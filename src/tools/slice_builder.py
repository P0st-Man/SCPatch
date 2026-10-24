from pathlib import Path
from typing import Dict, Optional
import re

class CodeSliceBuilder:
    def __init__(self, include_comments: bool = False):
        self.include_comments = include_comments

    def build_simplified_contract(
        self,
        func_context: Dict,
        fixed_code: Optional[str] = None
    ) -> str:
        contract_context = func_context.get('contract_context', {})
        contract_name = contract_context.get('contract_name', 'TestContract')

        def _get_code(item: object) -> str:
            if isinstance(item, dict):
                return str(item.get('code', '') or '')
            if isinstance(item, str):
                return item
            return ""

        def _sanitize_modifier_code(code: str) -> str:
            s = (code or "").strip()
            if not s:
                return ""
            if "modifier " in s or "{" in s or "}" in s:
                return code
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
                return f"// modifier: {s}"
            return code
        
        solidity_version = func_context.get('solidity_version', '0.8.0')
        
        full_contract = f"// SPDX-License-Identifier: MIT\n"
        full_contract += f"pragma solidity ^{solidity_version};\n\n"
        
        full_contract += f"contract {contract_name} {{\n"
        
        state_vars = contract_context.get('state_variables', [])
        if state_vars:
            full_contract += "    // State variables\n"
            for var in state_vars:
                var_code = _get_code(var)
                if var_code and not var_code.strip().startswith('//'):
                    clean_code = var_code.strip()
                    if not clean_code.endswith(';'):
                        clean_code += ';'
                    full_contract += "    " + clean_code + "\n"
            full_contract += "\n"
        
        structures = contract_context.get('structures', [])
        if structures:
            full_contract += "    // Structures\n"
            for struct in structures:
                struct_code = _get_code(struct)
                if struct_code:
                    for line in struct_code.split('\n'):
                        full_contract += "    " + line + "\n"
                    full_contract += "\n"

        modifiers = contract_context.get('modifiers', [])
        if modifiers:
            full_contract += "    // Modifiers\n"
            for mod in modifiers:
                mod_code = _sanitize_modifier_code(_get_code(mod))
                if mod_code:
                    for line in mod_code.split('\n'):
                        full_contract += "    " + line + "\n"
                    full_contract += "\n"
        
        
        target_code = fixed_code if fixed_code is not None else func_context.get('function_code', '')
        if target_code:
            full_contract += "    // Target function\n"
            for line in target_code.split('\n'):
                full_contract += "    " + line + "\n"
        
        full_contract += "}\n"
        return full_contract

    @staticmethod
    def _source_tokens(source):
        pattern = re.compile(
            r'//[^\r\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|'
            r"'(?:\\[\s\S]|[^'\\])*'|[A-Za-z_$][A-Za-z0-9_$]*|"
            r'0[xX][0-9a-fA-F_]+|[0-9][0-9_]*(?:\.[0-9_]+)?(?:[eE][+-]?[0-9_]+)?|'
            r'>>=|<<=|\*\*=|\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|\*\*|=>|'
            r'\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]'
        )
        return [m for m in pattern.finditer(source) if not m.group().startswith(('//', '/*'))]

    def rebuild_full_contract(self, func_data: Dict, fixed_code: str) -> str:
        if not isinstance(fixed_code, str) or not fixed_code.strip():
            raise ValueError('Repair code is empty')
        name = func_data.get('contract_name') or func_data.get('contract_context', {}).get('contract_name')
        if func_data.get('repair_scope') == 'contract':
            values = [m.group() for m in self._source_tokens(fixed_code)]
            pairs = list(zip(values, values[1:]))
            if not name or ('contract', name) not in pairs:
                raise ValueError('Contract-scoped repair must contain the complete target contract')
            if ('pragma', 'solidity') not in pairs:
                raise ValueError('Contract-scoped repair must include its Solidity pragma')
            return fixed_code

        path = func_data.get('contract_path') or func_data.get('metadata', {}).get('contract_file')
        if not path:
            raise ValueError('Original contract path is required for verification')
        with Path(path).open(encoding='utf-8', newline='') as stream:
            source = stream.read()
        original = [m.group() for m in self._source_tokens(func_data.get('function_code') or '')]
        if not original:
            raise ValueError('Original function code is required for verification')
        tokens = self._source_tokens(source)
        values = [m.group() for m in tokens]
        regions = []
        if name:
            for i in range(len(values) - 1):
                if values[i] not in ('contract', 'library', 'interface') or values[i + 1] != name:
                    continue
                begin = i + 2
                while begin < len(values) and values[begin] != '{':
                    begin += 1
                depth = 0
                for end in range(begin, len(values)):
                    depth += (values[end] == '{') - (values[end] == '}')
                    if depth == 0:
                        regions.append((begin, end))
                        break

        matches = []
        for i in range(len(values) - len(original) + 1):
            if values[i] != original[0] or values[i:i + len(original)] != original:
                continue
            last = i + len(original) - 1
            if name and not any(begin < i and last < end for begin, end in regions):
                continue
            matches.append((tokens[i].start(), tokens[last].end()))

        start_line, end_line = func_data.get('start_line'), func_data.get('end_line')
        hinted = [(start, end) for start, end in matches
                  if source[:start].count('\n') + 1 == start_line
                  and source[:end].count('\n') + 1 == end_line]
        if len(hinted) == 1:
            matches = hinted
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely locate original function in {path}: {len(matches)} matches')
        start, end = matches[0]
        return source[:start] + fixed_code + source[end:]
