import re
from pathlib import Path
import pandas as pd

"""
Normalize the `conversation` column into a consistent "Speaker: utterance" format.

Goal format per turn: "Speaker1: utterance Speaker2: utterance2 ..."

Heuristics used:
- Split conversation text into candidate turns using common separators (line breaks, long spaces, tabs, or numbered markers like '1.', '2.').
- For each candidate turn, try to extract the speaker by:
  - detecting a leading 'Name:' pattern
  - detecting a name followed by quotes or double spaces
  - if no speaker is detected, inherit the previous one or alternate between name1/name2 if available
- Build the normalized string by joining all turns in the form "Speaker: utterance".

This script outputs two CSV files (preserving all other columns):
1. `...(final_normalized_names).csv` — with original speaker names
2. `...(final_normalized_anon).csv` — with speaker names replaced by generic labels (Speaker1, Speaker2)
"""

SPLIT_RE = re.compile(r"\n|\r|\s{2,}|\t|(?<!\S)\d{1,3}\.\s+")

def split_speaker_utterance(s):
    # Try common patterns: Name: utterance
    s = s.strip()
    # If starts with a number and dot, remove it
    s = re.sub(r'^\d+\.\s*', '', s)

    # Pattern: Speaker: utterance
    m = re.match(r'^(?P<sp>[^:\-\n]{1,30})\s*:\s*(?P<ut>.*)$', s)
    if m:
        return m.group('sp').strip(), m.group('ut').strip()

    # Pattern: Speaker "..." Utterance or Speaker """
    m = re.match(r'^(?P<sp>\w[\w\s\-]{0,30})\s+"{1,3}(?P<ut>.*)$', s)
    if m:
        return m.group('sp').strip(), m.group('ut').strip()

    # Pattern: Name followed by message on same line separated by multiple spaces where name may be a token
    m = re.match(r'^(?P<sp>\S{1,30})\s{2,}(?P<ut>.*)$', s)
    if m:
        return m.group('sp').strip(), m.group('ut').strip()

    # If none matched, return (None, whole)
    return (None, s)

def normalize_conversation(conv, name1=None, name2=None):
    if not conv:
        return ''

    # normalize quotes
    text = conv.replace('""', '"')
    # Replace long multiple spaces used in the CSV as separators with newline
    text = re.sub(r' {3,}', '\n', text)
    # Ensure colon after speaker is normalized (e.g., 'Name:' or 'Name :')
    text = re.sub(r'\s+:', ':', text)

    # Remove standalone numbered turn prefixes like '1. ', '2.' etc.
    # Only remove numbers that act as standalone tokens followed by a dot and space(s).
    # This preserves decimals/times such as '22.50' (dot followed by digit).
    text = re.sub(r'(?<!\S)\d{1,3}\.(?=\s+)', '', text)

    # Split into candidate lines
    parts = [p.strip() for p in re.split(r'\n|\r', text) if p.strip()]

    turns = []
    last_speaker = None
    for p in parts:
        # If the part contains numbered items like '1. Name: text 2. Name2: text', split by numbered pattern
        # detect explicit enumerations like '1. ... 2. ...' (number dot followed by space)
        if re.search(r'(?<!\S)\d{1,3}\.\s+', p):
            # split on occurrences of digits + dot used as enumerators (preserve decimals)
            sub = re.split(r'(?<=\.)\s*(?=(?:\d{1,3}\.\s+))', p)
            for s in sub:
                s = s.strip()
                if not s:
                    continue
                sp, utt = split_speaker_utterance(s)
                if sp:
                    last_speaker = sp
                else:
                    # inherit last speaker when possible, otherwise leave None
                    sp = last_speaker if last_speaker is not None else None
                turns.append((sp, utt))
        else:
            sp, utt = split_speaker_utterance(p)
            if sp:
                last_speaker = sp
            else:
                sp = last_speaker if last_speaker is not None else None
            turns.append((sp, utt))

    # Build normalized string. Replace UNKNOWN with name1/name2 by parity if available.
    out_parts = []
    for i, (sp, utt) in enumerate(turns, start=1):
        # Clean speaker quotes
        sp = (sp or '').strip().strip('"')
        utt = utt.strip().strip('"')
        if not sp:
            # decide based on turn parity: odd -> name1, even -> name2
            if i % 2 == 1:
                sp = (name1 or '')
            else:
                sp = (name2 or '')
        # append without numbering
        out_parts.append(f"{sp}: {utt}")

    return ' '.join(out_parts)


if __name__ == '__main__':
    # Default input and outputs (edit if needed)
    input_file = Path('datasets/classification_and_explaination_toxic_conversation(merged_cleaned).csv')
    output_file_with_names = Path('datasets/classification_and_explaination_toxic_conversation(final_normalized_names).csv')
    output_file_anon = Path('datasets/classification_and_explaination_toxic_conversation(final_normalized_anon).csv')

    if not input_file.exists():
        raise SystemExit(f'Input file not found: {input_file}')

    df = pd.read_csv(input_file, encoding='utf-8').fillna('')

    rows = []
    rows_anon = []
    for _, r in df.iterrows():
        conv = r.get('conversation') or r.get('conversation ') or r.get(' conversation') or r.get(' Conversation') or ''
        name1 = r.get('name1') or r.get(' Name1') or r.get('name_1') or r.get('name') or ''
        name2 = r.get('name2') or r.get(' Name2') or r.get('name_2') or r.get('name2 ') or ''

        norm = normalize_conversation(conv, name1=name1, name2=name2)

        # create two copies: one with original names, one anonymized
        r_out = r.copy()
        r_out['conversation'] = norm
        rows.append(r_out)

        # anonymize names in conversation: replace occurrences of name1/name2 with Speaker labels
        anon_conv = norm
        if name1:
            anon_conv = re.sub(re.escape(name1), 'Speaker1', anon_conv)
        if name2:
            anon_conv = re.sub(re.escape(name2), 'Speaker2', anon_conv)
        # Also replace plain name fields
        r_anon = r.copy()
        r_anon['conversation'] = anon_conv
        # substitute name columns with generic labels
        r_anon['name1'] = 'Speaker1'
        r_anon['name2'] = 'Speaker2'
        rows_anon.append(r_anon)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_file_with_names, index=False, encoding='utf-8')

    out_anon_df = pd.DataFrame(rows_anon)
    out_anon_df.to_csv(output_file_anon, index=False, encoding='utf-8')

    print(f'Wrote {len(out_df)} rows to {output_file_with_names} and anonymized file to {output_file_anon}')
