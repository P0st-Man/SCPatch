from typing import Dict, List


class PromptFormatter:
    def format_fix_instruction(self) -> str:
        prompt = f"""You are an expert Solidity security engineer specializing in vulnerability remediation
# **Your Task:**
# Fix the vulnerabilities in the provided Solidity code. 
# Ensure the business logic remains unchanged.

# **Output Format:**
# Return ONLY the corrected Solidity function code in a code block not the entire contract.
# """
        prompt = f"""You are an expert Solidity smart contract security auditor. 
Your task is to fix security vulnerabilities in Solidity code while maintaining functionality. 
Provide the complete fixed function code.
"""
        return prompt

    def format_general_fix_prompt(self, code_slice: str, annotation: Dict, solc_version: str) -> str:
        vulnerability_types = annotation.get('vulnerability_types', [])
        analysis = annotation.get('analysis', '')
        severity = annotation.get('severity', 0)
        
        prompt = f"""You are an expert Solidity security engineer. Your task is to fix the vulnerabilities in the following smart contract function.

**Original Vulnerable Code:**
```solidity
{code_slice}
```

**Security Analysis:**
{analysis}

**Identified Vulnerabilities:**
{', '.join(vulnerability_types)}

**Severity Score:** {severity}/10

**Your Task:**
Generate a FIXED version of this function that:
1. **Eliminates all identified vulnerabilities**
2. **Maintains the original functionality** (same inputs/outputs, same business logic)
3. **Follows Solidity best practices**
4. **Is syntactically correct** and will compile
5. **Uses appropriate Solidity version features, currently using Solidity {solc_version}**
"""
        return prompt

    def format_general_fix_prompt_rich(self, code_slice: str, annotation: Dict, solc_version: str) -> str:
        vulnerability_types = annotation.get('vulnerability_types', [])
        analysis = annotation.get('analysis', '')
        severity = annotation.get('severity', 0)

        vulnerable_code_details = annotation.get("vulnerable_code_details") or ""
        slither_section = annotation.get("slither_section") or ""

        prompt = f"""You are an expert Solidity security engineer. Your task is to fix the vulnerabilities in the following smart contract function.

**Original Vulnerable Code:**
```solidity
{code_slice}
```

## Ground Truth (Reference)
This function is labeled as vulnerable, with vulnerability types: {', '.join(vulnerability_types)}.

{vulnerable_code_details}

{slither_section}

## Security Context
{analysis}

## Your Task
Generate a FIXED version of this function that:
1. **Eliminates all identified vulnerabilities**
2. **Maintains the original functionality** (same inputs/outputs, same business logic)
3. **Follows Solidity best practices**
4. **Is syntactically correct** and will compile
5. **Uses appropriate Solidity version features, currently using Solidity {solc_version}**
"""
        return prompt

    def format_general_system_prompt(self) -> str:
        return """You are an expert Solidity security engineer specializing in vulnerability remediation.
**CRITICAL Requirements:**
1. Return ONLY the fixed function code (NOT the entire contract)
2. Preserve the function signature (name, parameters, return types, visibility)
3. Keep the same state variable interactions whenever possible
4. Ensure the code is complete and will compile
5. Add brief security comments explaining key fixes

**Required Output (JSON format):**
{{
  "fixed_code": "The complete fixed function code in Solidity",
  "fix_analysis": "A clear paragraph explaining what vulnerabilities were fixed and how. Describe the specific changes made and why they improve security.",
  "key_changes": ["list", "of", "main", "security", "improvements"],
  "solidity_version_notes": "Any version-specific considerations (e.g., SafeMath vs built-in checks)"
}}    
"""

    def format_retry_prompt(self, previous_fixed_code: str, error_summary: str, vulnerability_types: List[str], solc_version: str) -> str:
        prompt = f"""You are an expert Solidity security engineer. Your previous fix attempt failed to compile.

**Solidity Version: {solc_version}**

**Previous Fix Attempt (Function Code):**
```solidity
{previous_fixed_code}
```

**Compilation Error:**
```
{error_summary}
```

**Vulnerabilities to Fix:**
{', '.join(vulnerability_types)}

**Your Task:**
Fix the compilation error in the function above while still addressing all vulnerabilities.

**Output Format:**
Return ONLY the corrected Solidity function code in a code block.

**Example:**
```solidity
function fixedFunction() public {{
    // Fixed code
}}
```"""
        return prompt

    def format_retry_system_prompt(self) -> str:
        return """You are an expert Solidity security engineer specializing in fixing compilation errors.
**CRITICAL Requirements for Retry:**
1. Return ONLY the fixed function code (NOT the entire contract)
2. Preserve ALL security fixes from the previous attempt
3. Fix ONLY the compilation errors while keeping everything else unchanged
4. Ensure the code is complete and will compile
5. Do NOT add any explanations, JSON, or other text - just the code

**Required Output for Retry:**
```solidity
// The complete fixed function code ONLY
function functionName(...) ... {
    // Fixed code here
}
```"""

    def format_fix_prompt_for_our_models(self, code_slice: str, annotation: Dict, solc_version: str, function_name: str, include_instruction: bool = False) -> str:
        annotation = annotation or {}
        vulnerability_types = annotation.get('vulnerability_types', [])
        severity = annotation.get('severity', 0)
        
        meta_lines = []
        
        if vulnerability_types:
            if isinstance(vulnerability_types, list):
                types = ", ".join(vulnerability_types)
            else:
                types = str(vulnerability_types)
            meta_lines.append(f"- Type: {types}")
        
        if severity is not None:
            meta_lines.append(f"- Severity: {severity}")
        
        input_text = f"The following Solidity function has been identified with security vulnerabilities: \n\n"
        input_text += f"**Function:** {function_name}\n\n"
        if meta_lines:
            input_text += "**Vulnerability Info:**\n" + "\n".join(meta_lines) + "\n\n"
        
        input_text += f"**Source Code:**\n{code_slice}"

        input_text += f"Please provide the complete fixed version of this code that addresses the identified vulnerabilities. \nOutput ONLY the corrected Solidity function code in a code block not the entire contract."
        if include_instruction:
            instruction = self.format_fix_instruction()
            full_prompt = f"{instruction}\n\n{input_text}"
            return full_prompt
        else:
            return input_text
