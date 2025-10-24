import re
from pathlib import Path
import pandas as pd

"""
Repair failures from the `...(failures).csv` produced earlier and combine with cleaned CSV.

The script applies a right-split heuristic to extract name1,name2,explaination,toxic
from the original broken line and extracts person_couple and conversation from the
remaining prefix. It writes repaired rows and a combined final CSV.
"""

def normalize(s):
    if s is None:
        return ''
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = s.replace('""', '"')
    s = re.sub(r'\s+', ' ', s).strip()
    # strip surrounding quotes
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def repair_row(orig):
    # try to rsplit into prefix, name1, name2, explain, toxic
    parts = orig.rsplit(',', 4)
    if len(parts) == 5:
        prefix, name1, name2, explain, toxic = parts
    else:
        # fallback: rsplit 3
        parts = orig.rsplit(',', 3)
        if len(parts) == 4:
            prefix, name1, name2, toxic = parts
            explain = ''
        else:
            return None

    prefix = normalize(prefix)
    name1 = normalize(name1)
    name2 = normalize(name2)
    explain = normalize(explain)
    toxic = normalize(toxic)

    # Sometimes toxic sits in explain and ends with Sì/No at the very end; try to split
    # If toxic looks like 'Sì' or 'Si' or 'No' keep, else try to extract last token
    if toxic.lower() not in ('sì', 'si', 'no', 'yes', 'sí'):
        # attempt to pull a final token
        m = re.search(r"(.*),\s*([Ss]i|Sì|No)\s*$", orig)
        if m:
            toxic = m.group(2)
            explain = normalize(m.group(1))

    # Extract person_couple as first token before first comma
    if ',' in prefix:
        first, rest = prefix.split(',', 1)
        person = normalize(first)
        conversation = normalize(rest)
    else:
        # if no comma, try to take initial quoted block
        m = re.match(r'^"?([^\"]+)"?\s*(.*)$', prefix)
        if m:
            person = normalize(m.group(1))
            conversation = normalize(m.group(2))
        else:
            person = prefix
            conversation = ''

    # Ensure person contains ' e ' or fallback to first words
    if ' e ' not in person and ' e,' in person:
        person = person.replace(' e,', ' e ')

    return {
        'person_couple': person,
        'conversation': conversation,
        'name1': name1,
        'name2': name2,
        'explaination': explain,
        'toxic': toxic,
        'orig': orig,
    }


if __name__ == '__main__':
    failures_file = Path('datasets/classification_and_explaination_toxic_conversation(failures).csv')
    cleaned_file = Path('datasets/classification_and_explaination_toxic_conversation(cleaned).csv')
    output_file = Path('datasets/classification_and_explaination_toxic_conversation(merged_cleaned).csv')

    if not failures_file.exists():
        raise SystemExit(f'Failures file not found: {failures_file}')

    repaired = []
    still_failed = []

    df_fail = pd.read_csv(failures_file, encoding='utf-8').fillna('')
    for _, row in df_fail.iterrows():
        # handle possible column name variants
        orig = row.get('orig', '') or row.get('orig ', '') or ''
        if not orig:
            still_failed.append((row.get('line_number', ''), 'empty_orig'))
            continue
        res = repair_row(orig)
        if res:
            repaired.append(res)
        else:
            still_failed.append((row.get('line_number', ''), str(orig)[:120]))

    # Combine: existing + repaired
    combined = []
    if cleaned_file.exists():
        df_existing = pd.read_csv(cleaned_file, encoding='utf-8').fillna('')
        for _, r in df_existing.iterrows():
            combined.append({
                'person_couple': normalize(r.get('person_couple', '')),
                'conversation': normalize(r.get('conversation', '')),
                'name1': normalize(r.get('name1', '')),
                'name2': normalize(r.get('name2', '')),
                'explaination': normalize(r.get('explaination', '')),
                'toxic': normalize(r.get('toxic', '')),
            })

    for r in repaired:
        combined.append({
            'person_couple': normalize(r['person_couple']),
            'conversation': normalize(r['conversation']),
            'name1': normalize(r['name1']),
            'name2': normalize(r['name2']),
            'explaination': normalize(r['explaination']),
            'toxic': normalize(r['toxic']),
        })

    # Write combined to out_path using pandas
    out_df = pd.DataFrame(combined)
    # Ensure columns order
    cols = ['person_couple', 'conversation', 'name1', 'name2', 'explaination', 'toxic']
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ''
    out_df = out_df[cols]
    out_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f'Repaired {len(repaired)} rows, still failed: {len(still_failed)}')
    if still_failed:
        print('Sample remaining failures (line_number, excerpt):')
        for lf in still_failed[:10]:
            print(lf)

    print('Final combined CSV written to', output_file)