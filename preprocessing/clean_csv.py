import re
from pathlib import Path
import pandas as pd

def clean_line(line):
    """
    Heuristic cleaner for corrupted CSV where most data ended up in the first quoted field.
    Strategy (heuristic):
    - Work line-by-line (input appears as one-record-per-line in the broken file).
    - Find the start of the explanation field by searching from the right for the sequence ',""' or ',"'.
    - Split the tail into explanation and toxic (last comma separates toxic label).
    - From the prefix (everything before explaination start) split by commas and take the last two pieces as name1,name2.
    - Try to extract `person_couple` from the start of the line by matching a short pattern containing ' e '. If not found, use the first comma-separated piece.
    - Conversation is the remainder between person_couple and the two names.
    """
    orig = line
    # Quick guard
    if not line.strip():
        return None

    # Find explanation start
    idx = line.rfind(',""')
    if idx == -1:
        idx = line.rfind(',"')
    if idx == -1:
        # fallback: try to find '"La conversazione' or '"Tossica' heuristic
        idx = line.rfind(',"La conversazione')
    if idx == -1:
        # give up this heuristic for this line
        return {'error': 'no_explain_start', 'orig': orig}

    prefix = line[:idx]
    tail = line[idx+1:]

    # split explanation and toxic by last comma
    if ',' in tail:
        explain_part, toxic_part = tail.rsplit(',', 1)
    else:
        explain_part = tail
        toxic_part = ''

    explain = explain_part.strip().strip('"').replace('""', '"').strip()
    toxic = toxic_part.strip().strip('"').strip()
    # normalize toxic labels
    if toxic.lower() in ('si', 'sì'):
        toxic = 'Sì'
    elif toxic.lower() in ('no', 'nò'):
        toxic = 'No'

    # From prefix, split by comma and take last two tokens as names (heuristic)
    parts = prefix.split(',')
    if len(parts) < 3:
        return {'error': 'not_enough_parts', 'orig': orig, 'prefix': prefix}

    name1 = parts[-2].strip().strip('"')
    name2 = parts[-1].strip().strip('"')

    # Try to extract person_couple from the beginning (look for ' e ' pattern)
    m = re.match(r'^"([^,]{1,80}\se\s[^,\"]{1,80})', line)
    if m:
        person = m.group(1).strip()
    else:
        # fallback: first comma-separated piece
        person = parts[0].strip().strip('"')

    # conversation is everything between person_couple and the two names
    if len(parts) > 3:
        conv_parts = parts[1:-2]
        conversation = ",".join(conv_parts).strip().strip('"').replace('""', '"')
    else:
        conversation = ''

    # Normalize whitespace and quotes
    conversation = re.sub(r'\s+', ' ', conversation).strip()
    explain = re.sub(r'\s+', ' ', explain).strip()
    person = person.strip()
    name1 = name1.strip()
    name2 = name2.strip()

    return {
        'person_couple': person,
        'conversation': conversation,
        'name1': name1,
        'name2': name2,
        'explaination': explain,
        'toxic': toxic,
        'orig': orig
    }

if __name__ == '__main__':
    input_file = Path('datasets/classification_and_explaination_toxic_conversation(in).csv')
    output_file = Path('datasets/classification_and_explaination_toxic_conversation(cleaned).csv')

    if not input_file.exists():
        raise SystemExit(f'Input file not found: {input_file}')
    # Read the file contents and create a pandas DataFrame of lines.
    # pandas doesn't allow '\n' as a separator with the python engine, so read text
    # and wrap into a DataFrame instead.
    text = input_file.read_text(encoding='utf-8')
    lines = text.splitlines()
    df = pd.DataFrame({'raw_line': [str(x) for x in lines]})
    if df.empty:
        raise SystemExit('Input file is empty')

    header = df.iloc[0]['raw_line']
    cleaned = []
    failures = []

    for i, line in enumerate(df['raw_line'].iloc[1:], start=2):
        # ensure we pass a string
        line = line if isinstance(line, str) else ''
        res = clean_line(line)
        if not res:
            continue
        if 'error' in res:
            failures.append((i, res))
        else:
            cleaned.append(res)

    # Write cleaned CSV
    out_df = pd.DataFrame(cleaned)
    # Ensure columns order
    cols = ['person_couple', 'conversation', 'name1', 'name2', 'explaination', 'toxic']
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ''
    out_df = out_df[cols]
    out_df.to_csv(output_file, index=False, encoding='utf-8')

    # Write failures to a separate file for manual inspection
    if failures:
        failed_path = input_file.with_name(input_file.name.replace('(in)', '(failures)'))
        failures_rows = []
        for ln, info in failures:
            failures_rows.append({
                'line_number': ln,
                'error': info.get('error'),
                'prefix': info.get('prefix', ''),
                'orig': info.get('orig', '')
            })
        failures_df = pd.DataFrame(failures_rows)
        failures_df.to_csv(failed_path, index=False, encoding='utf-8')
        print('Failures written to:', failed_path)

    # Print brief report
    print(f'Read {len(df)-1} data lines')
    print(f'Parsed {len(cleaned)} records, failures: {len(failures)}')
    if failures:
        print('Sample failures:')
        for i, fdata in failures[:5]:
            print(i, fdata.get('error'), (fdata.get('prefix') or '')[:120])

    print('\nOutput written to:', output_file)
