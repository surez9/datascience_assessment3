"""
Step 0 - Geting the venue data from public sources (nothing is typed in by hand).
"""
import sys
from pathlib import Path

import geonamescache
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCE = sys.argv[1] if len(sys.argv) > 1 else \
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# A few countries are spelled differently in the two sources.
TEAM_NAMES = {"USA": "United States", "Czechia": "Czech Republic", "Türkiye": "Turkey",
              "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Cabo Verde": "Cape Verde",
              "Côte d'Ivoire": "Ivory Coast"}

# 1. The 104 World Cup 2026 matches from the public results list
results = pd.read_csv(SOURCE, parse_dates=["date"])
results = results[(results["tournament"] == "FIFA World Cup") & (results["date"] >= "2026-06-01")]
results["pair"] = [frozenset(p) for p in zip(results["home_team"], results["away_team"])]

# 2. Your matches: the two team names identify the match, and venue_id tells us the venue
details = pd.read_csv(RAW / "post_match_details.csv")[["event_id", "venue_id", "home_team", "away_team"]]
details["pair"] = [frozenset(TEAM_NAMES.get(t, t) for t in p) for p in zip(details["home_team"], details["away_team"])]
matches = details.merge(results[["pair", "city", "country"]], on="pair", how="left", validate="1:1")
assert len(matches) == 104 and matches["city"].notna().all(), "some matches could not be matched"

# 3. One city and country per venue_id (the most common one if the sources disagree)
venues = matches.groupby("venue_id").agg(city=("city", lambda s: s.mode()[0]),
                                         country=("country", lambda s: s.mode()[0])).reset_index()

# 4. Latitude and longitude of each city from GeoNames
geo = geonamescache.GeonamesCache(min_city_population=500)
country_code = {c["name"]: code for code, c in geo.get_countries().items()}


def find_city(name, country):
    """Largest city with exactly this name in the country (alternative names only if there is no exact match)."""
    in_country = [c for c in geo.get_cities().values() if c["countrycode"] == country_code[country]]
    found = [c for c in in_country if c["name"] == name] or [c for c in in_country if name in c["alternatenames"]]
    return max(found, key=lambda c: c["population"])


places = [find_city(city, country) for city, country in zip(venues["city"], venues["country"])]
venues["latitude"] = [round(p["latitude"], 3) for p in places]
venues["longitude"] = [round(p["longitude"], 3) for p in places]

venues.to_csv(RAW / "venues.csv", index=False)
print(venues.to_string(index=False))
