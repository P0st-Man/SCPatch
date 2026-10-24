from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import contextmanager

try:
    from slither.slither import Slither
    SLITHER_AVAILABLE = True
except ImportError:
    print("Warning: Slither not available.")
    SLITHER_AVAILABLE = False

try:
    import solcx
    SOLCX_AVAILABLE = True
except ImportError:
    print("Warning: py-solc-x not available. Version switching disabled.")
    SOLCX_AVAILABLE = False


class SlitherManager:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._version_cache = {}
        self._old_env = {}
        
    def setup_solc_version(self, contract_file: str) -> str:
        if not SOLCX_AVAILABLE:
            raise RuntimeError('py-solc-x is required for compiler selection')
        from semantic_version import NpmSpec, Version
        from src.tools.slice_builder import CodeSliceBuilder

        source = Path(contract_file).read_text(encoding='utf-8')
        tokens = CodeSliceBuilder._source_tokens(source)
        declarations = []
        for i, token in enumerate(tokens[:-2]):
            if token.group() != 'pragma' or tokens[i + 1].group() != 'solidity':
                continue
            end = i + 2
            while end < len(tokens) and tokens[end].group() != ';':
                end += 1
            if end == len(tokens):
                raise ValueError('Unterminated Solidity pragma')
            declaration = source[tokens[i + 2].start():tokens[end].start()]
            declaration = re.sub(r'/\*[\s\S]*?\*/|//[^\r\n]*', ' ', declaration)
            declaration = re.sub(r'([<>=^~]+)\s+', r'\1', declaration).strip()
            declarations.append(declaration)
        if not declarations:
            raise ValueError('A Solidity pragma is required for compiler selection')
        key = (tuple(declarations), source)
        cached = self._version_cache.get(key)
        if cached and os.path.isfile(cached):
            return cached
        specs = [NpmSpec(value) for value in declarations]

        def compatible(versions):
            parsed = [Version(str(v).lstrip('v')) for v in versions]
            return sorted((v for v in parsed if all(spec.match(v) for spec in specs)), reverse=True)

        matches = compatible(solcx.get_installed_solc_versions())
        if not matches:
            matches = compatible(solcx.get_installable_solc_versions())
            if not matches:
                raise ValueError('No compiler satisfies Solidity pragmas: ' + '; '.join(declarations))
            solcx.install_solc(str(matches[0]))
            matches = matches[:1]
        errors = []
        deadline = time.monotonic() + 30
        for version in matches:
            executable = str(solcx.install.get_executable(str(version)))
            if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
                errors.append(f'{version}: compiler is not executable')
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Solidity compiler selection timed out')
            compiled = subprocess.run([executable, '--bin', contract_file],
                                      capture_output=True, text=True, timeout=remaining)
            if compiled.returncode == 0:
                if len(self._version_cache) >= 32:
                    self._version_cache.pop(next(iter(self._version_cache)))
                self._version_cache[key] = executable
                return executable
            errors.append(f'{version}: {compiled.stderr[-2000:]}')
        raise ValueError('Source did not compile with installed compatible compilers:\n' + '\n'.join(errors))

    def _setup_environment(self, solc_path: str):
        env_vars_to_save = ['SOLC', 'SOLC_VERSION', 'PATH']
        for var in env_vars_to_save:
            self._old_env[var] = os.environ.get(var)
        
        solc_select_vars = ['SOLC_SELECT_VERSION', 'SOLC_SELECT', 'SOLC_SELECT_DIR']
        for var in solc_select_vars:
            if var in os.environ:
                del os.environ[var]
        
        if os.path.exists(solc_path):
            os.environ['SOLC'] = solc_path
            solc_dir = os.path.dirname(solc_path)
            old_path = os.environ.get('PATH', '')
            os.environ['PATH'] = f"{solc_dir}:{old_path}"
    
    def _restore_environment(self):
        for var, value in self._old_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
        self._old_env.clear()
    
    def get_slither(self, 
                    contract_file: str,
                    **kwargs) -> Optional[Slither]:
        if not SLITHER_AVAILABLE:
            print("Error: Slither not available. Please install: pip install slither-analyzer")
            return None
        
        if not os.path.exists(contract_file):
            print(f"Error: Contract file not found: {contract_file}")
            return None
        
        try:
            if self.debug:
                print(f"[1/4] Setting up compiler version for {os.path.basename(contract_file)}...")
            
            solc_path = self.setup_solc_version(contract_file)
            
            if not solc_path:
                print(f"Error: Failed to setup compiler version for {contract_file}")
                print("Possible reasons:")
                print("  - No pragma statement found")
                print("  - py-solc-x not installed: pip install py-solc-x")
                print("  - Compiler version not available")
                return None
            
            if self.debug:
                print(f"    ✓ Compiler setup complete: {solc_path}")
            
            if self.debug:
                print(f"[2/4] Setting up environment variables...")
            
            self._setup_environment(solc_path)
            
            if self.debug:
                print(f"    ✓ Environment configured")
            
            if self.debug:
                print(f"[3/4] Preparing Slither parameters...")
            
            slither_kwargs = {
                'solc_disable_warnings': True,
            }
            
            if os.path.exists(solc_path):
                slither_kwargs['solc'] = solc_path
                if self.debug:
                    print(f"    ✓ Using solc: {solc_path}")
            
            slither_kwargs.update(kwargs)
            
            if self.debug:
                print(f"[4/4] Creating Slither instance...")
            
            slither = Slither(contract_file, **slither_kwargs)
            
            if self.debug:
                if slither:
                    print(f"    ✓ Slither instance created successfully")
                    print(f"    ✓ Found {len(slither.contracts)} contract(s)")
                else:
                    print(f"    ✗ Slither instance creation failed")
            
            return slither
        
        except FileNotFoundError as e:
            print(f"Error: File not found: {e}")
            self._restore_environment()
            return None
        
        except Exception as e:
            print(f"Error creating Slither instance for {contract_file}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            
            if "solc" in str(e).lower():
                print(f"  Hint: This might be a compiler issue")
                print(f"       - Check if the contract can compile manually")
                print(f"       - Try: solc {contract_file}")
            
            if self.debug:
                print("\nFull traceback:")
                import traceback
                traceback.print_exc()
            
            self._restore_environment()
            return None
    
    @contextmanager
    def analyze_contract(self, 
                        contract_file: str,
                        **kwargs):
        slither = None
        try:
            slither = self.get_slither(contract_file, **kwargs)
            yield slither
        except GeneratorExit:
            pass
        except Exception as e:
            if self.debug:
                print(f"Exception in analyze_contract: {e}")
                import traceback
                traceback.print_exc()
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        self._restore_environment()
