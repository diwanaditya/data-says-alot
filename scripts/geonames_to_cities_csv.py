from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
from typing import Dict, Optional


def open_maybe_gz(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    return open(path, 'r', encoding='utf-8', errors='ignore')


def load_admin1_map(admin1_path: str) -> Dict[str, str]:
    """Load GeoNames admin1CodesASCII.txt into a mapping of code -> name.

    Lines in admin1CodesASCII.txt look like:
      IN.27\tMaharashtra\tMumbai (Bombay)\t... (tab-separated)

    We map the left 'IN.27' -> 'Maharashtra'.
    """
    m: Dict[str, str] = {}
    with open(admin1_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            code = parts[0]
            name = parts[1] if len(parts) > 1 else ''
            m[code] = name
    return m


def convert_geonames_to_csv(input_path: str, output_path: str, country: str = 'IN', min_pop: int = 0,
                            admin1_map_path: Optional[str] = None, prefer_ascii: bool = True) -> Dict[str, int]:
    """Convert GeoNames TSV to CSV filtered by country and population.

    Returns a dict with stats.
    """
    admin_map = load_admin1_map(admin1_map_path) if admin1_map_path else {}

    total = 0
    written = 0
    try:
        with open_maybe_gz(input_path) as infp, open(output_path, 'w', encoding='utf-8', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(['name', 'admin1', 'country', 'latitude', 'longitude', 'population', 'geonameid'])
            for line in infp:
                total += 1
                line = line.rstrip('\n')
                if not line:
                    continue
                fields = line.split('\t')
                if len(fields) < 15:
                    # unexpected format; skip
                    continue
                geonameid = fields[0]
                name = fields[2] if prefer_ascii and fields[2] else fields[1]
                lat = fields[4]
                lon = fields[5]
                country_code = fields[8]
                admin1_code = fields[10]
                pop_str = fields[14] if fields[14] else '0'
                try:
                    pop = int(pop_str)
                except Exception:
                    pop = 0
                if country_code.upper() != country.upper():
                    continue
                if pop < int(min_pop):
                    continue
                admin1_name = ''
                if admin1_code:
                    full_code = f"{country.upper()}.{admin1_code}"
                    admin1_name = admin_map.get(full_code, admin1_code)
                writer.writerow([name, admin1_name, country_code.upper(), lat, lon, str(pop), geonameid])
                written += 1
    except FileNotFoundError as e:
        print(f"Error: file not found: {e}")
        raise

    return {'total_lines': total, 'written': written}


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description='Convert GeoNames dump to compact cities CSV (India default)')
    p.add_argument('--input', '-i', required=True, help='Path to GeoNames TSV dump (e.g., IN.txt or allCountries.txt or IN.txt.gz)')
    p.add_argument('--output', '-o', default='cities.csv', help='Output CSV path (default: cities.csv)')
    p.add_argument('--country', '-c', default='IN', help='Country code filter (default: IN)')
    p.add_argument('--min-pop', type=int, default=0, help='Minimum population to include (default: 0)')
    p.add_argument('--admin1', help='Optional admin1CodesASCII.txt path to map admin1 codes to names')
    p.add_argument('--prefer-ascii', action='store_true', help='Prefer asciiname (3rd column) over name (2nd column)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_cli(argv)
    print('Input:', args.input)
    print('Output:', args.output)
    print('Country filter:', args.country)
    print('Min population:', args.min_pop)
    if args.admin1:
        print('Using admin1 map:', args.admin1)
    stats = convert_geonames_to_csv(args.input, args.output, country=args.country, min_pop=args.min_pop,
                                    admin1_map_path=args.admin1, prefer_ascii=args.prefer_ascii)
    print('Done. Wrote', stats['written'], 'records (out of', stats['total_lines'], 'lines scanned)')


if __name__ == '__main__':
    main()
