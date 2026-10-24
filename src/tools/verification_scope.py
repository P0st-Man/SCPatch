import json
import re

DETECTORS = {
    'REENTRANCY': {'reentrancy-eth', 'reentrancy-no-eth', 'reentrancy-benign', 'reentrancy-events', 'reentrancy-unlimited-gas'},
    'ACCESS_CONTROL': {'arbitrary-send-eth', 'arbitrary-send-erc20', 'arbitrary-send-erc20-permit', 'suicidal', 'tx-origin', 'controlled-delegatecall', 'unprotected-upgrade', 'protected-vars'},
    'ARITHMETIC': {'divide-before-multiply', 'incorrect-exp', 'incorrect-shift', 'tautology', 'type-based-tautology'},
    'UNCHECKED_CALL': {'unchecked-transfer', 'unchecked-lowlevel', 'unchecked-send', 'unused-return'},
    'BAD_RANDOMNESS': {'weak-prng'},
    'TIMESTAMP_DEPENDENCE': {'timestamp'},
    'DOS': {'calls-loop', 'costly-loop', 'locked-ether', 'return-bomb'},
    'STORAGE': {'uninitialized-storage', 'uninitialized-state', 'uninitialized-local', 'mapping-deletion', 'array-by-reference', 'storage-array'},
}
SWCS = {
    'REENTRANCY': {'107'}, 'ACCESS_CONTROL': {'105', '106', '112', '115'},
    'ARITHMETIC': {'101'}, 'UNCHECKED_CALL': {'104'}, 'BAD_RANDOMNESS': {'120'},
    'TIMESTAMP_DEPENDENCE': {'116'}, 'DOS': {'113', '126', '128'},
    'STORAGE': {'109', '124'},
}


def target_types(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = value.split(',')
    return sorted({str(v).strip().upper() for v in (value or []) if v})


def make_scope(func_data, source, repaired_function):
    types = target_types(func_data.get('vulnerability_types'))
    if not types:
        raise ValueError('Target vulnerability types are required for scoped verification')
    start = source.find(repaired_function)
    if start < 0 or source.find(repaired_function, start + 1) >= 0:
        raise ValueError('Cannot uniquely locate repaired function in full contract')
    return {'types': types, 'start_line': source[:start].count('\n') + 1,
            'end_line': source[:start + len(repaired_function.rstrip())].count('\n') + 1,
            'function': func_data.get('function_name'), 'contract': func_data.get('contract_name')}


def _category_matches(identifier, types, mapping):
    if any(t not in mapping for t in types):
        return True
    return identifier in set().union(*(mapping[t] for t in types))


def _intersects(lines, scope):
    regions = [(scope['start_line'], scope['end_line'])] + scope.get('callee_regions', [])
    return any(start <= int(line) <= end for line in lines for start, end in regions)


def slither_relevant(item, scope):
    if scope is None:
        return True
    if not _category_matches(str(item.get('check', '')), scope['types'], DETECTORS):
        return False
    located = [e.get('source_mapping', {}).get('lines', []) for e in item.get('elements', [])
               if e.get('type') in ('node', 'function')]
    located = [lines for lines in located if lines]
    return not located or any(_intersects(lines, scope) for lines in located)


def mythril_relevant(item, scope):
    if scope is None:
        return True
    swc = str(item.get('swc-id') or item.get('swcID') or item.get('swc_id') or '').replace('SWC-', '')
    if swc and not _category_matches(swc, scope['types'], SWCS):
        return False
    lines = [item['lineno']] if item.get('lineno') else []
    for loc in item.get('locations', []) or []:
        if loc.get('lineno') or loc.get('line'):
            lines.append(loc.get('lineno') or loc.get('line'))
    return not lines or _intersects(lines, scope)


def include_internal_callees(slither, scope):
    if scope is None:
        return
    regions = set()
    for contract in getattr(slither, 'contracts', []):
        if scope.get('contract') and contract.name != scope['contract']:
            continue
        for function in getattr(contract, 'functions_declared', []):
            lines = getattr(getattr(function, 'source_mapping', None), 'lines', [])
            if not lines or not (scope['start_line'] <= min(lines) <= scope['end_line']):
                continue
            for callee in function.all_internal_calls():
                callee_lines = getattr(getattr(callee, 'source_mapping', None), 'lines', [])
                if callee_lines:
                    regions.add((min(callee_lines), max(callee_lines)))
    scope['callee_regions'] = sorted(regions)
