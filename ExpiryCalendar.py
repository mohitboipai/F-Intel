"""
ExpiryCalendar.py
=================
Derives the actual NIFTY weekly expiry schedule from bhavcopy data.

The expiry weekday changed from Thursday to Wednesday during 2024.
We do NOT hardcode the cutover date -- we detect it by scanning observed
expiry dates from the already-loaded BhavCopyEngine data.

Public API
----------
    cal = ExpiryCalendar(bhav_engine)
    cal.build()                            # detect switchover from data

    expiry = cal.get_expiry_for_entry(entry_date)
    dte    = cal.get_dte(entry_date, expiry)
    window = cal.get_entry_window(expiry)  # (first_entry_day, last_entry_day)
    stats  = cal.weekday_stats()           # {"Thursday": n, "Wednesday": n, ...}
"""

import os
import sys
from datetime import date, datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Human-readable weekday names
_WD_NAMES = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
             3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}


def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()


class ExpiryCalendar:
    """
    Builds and queries the NIFTY weekly expiry schedule derived from
    actual bhavcopy expiry dates. Handles the Thursday -> Wednesday
    weekday switch automatically.
    """

    def __init__(self, bhav_engine):
        self._bhav = bhav_engine
        self._expiries: list[date] = []       # sorted list of all known expiry dates
        self._switchover: date | None = None  # first date using the NEW weekday
        self._pre_weekday:  int = 3           # Thursday (default pre-switch)
        self._post_weekday: int = 2           # Wednesday (default post-switch)
        self._built = False

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, start_date=None, end_date=None):
        """
        Scan all expiry dates from BhavCopyEngine and detect the weekday
        switchover. Call this after bhav_engine.load_range() has been run.
        """
        if start_date and end_date:
            raw = self._bhav.get_expiries(start_date, end_date)
        else:
            if not self._bhav.get_all_dates():
                return
            all_dates = self._bhav.get_all_dates()
            raw = self._bhav.get_expiries(all_dates[0], all_dates[-1])

        if not raw:
            self._built = True
            return

        # Keep only weekly expiries (exclude monthly/quarterly -- within 10 days of prior expiry)
        # All NSE F&O expiries are weekly or monthly; weekly ones are more frequent
        weekly = []
        for exp in sorted(raw):
            if not weekly:
                weekly.append(exp)
            else:
                gap = (exp - weekly[-1]).days
                if gap <= 10:      # within 10 days -> weekly
                    weekly.append(exp)
                else:
                    # large gap = monthly/quarterly -- skip unless no prior weekly
                    weekly.append(exp)   # keep all and let weekday detection handle

        self._expiries = sorted(set(weekly))

        # Count weekdays
        wd_counts = Counter(e.weekday() for e in self._expiries)
        sorted_wds = sorted(wd_counts.items(), key=lambda x: -x[1])

        if len(sorted_wds) >= 2:
            self._pre_weekday  = sorted_wds[0][0]
            self._post_weekday = sorted_wds[1][0]

            # Find switchover: first date where the weekday changes from the dominant one
            dominant = sorted_wds[0][0]
            minority = sorted_wds[1][0]

            # The switchover is the date of the first minority-weekday expiry
            minority_dates = [e for e in self._expiries if e.weekday() == minority]
            if minority_dates:
                self._switchover = min(minority_dates)

                # Determine which weekday is PRE and which is POST by chronological order
                dominant_before_switch = [
                    e for e in self._expiries
                    if e < self._switchover and e.weekday() == dominant
                ]
                if dominant_before_switch:
                    self._pre_weekday  = dominant
                    self._post_weekday = minority

        self._built = True
        print(f'[ExpiryCalendar] Built: {len(self._expiries)} expiries | '
              f'Pre-switch: {_WD_NAMES[self._pre_weekday]} | '
              f'Post-switch: {_WD_NAMES[self._post_weekday]} | '
              f'Switchover: {self._switchover}')

    # ── expiry lookup ─────────────────────────────────────────────────────────

    def _expected_weekday(self, d: date) -> int:
        """Return the expected expiry weekday for a given date."""
        if self._switchover is None or d < self._switchover:
            return self._pre_weekday
        return self._post_weekday

    def get_expiry_for_entry(self, entry_date) -> date | None:
        """
        Return the nearest weekly expiry on or after entry_date.
        1. First tries the known expiry list from bhavcopy.
        2. Falls back to synthesising the next expected weekday.
        """
        entry = _to_date(entry_date)

        # Look in known expiry list
        future = [e for e in self._expiries if e >= entry]
        if future:
            return min(future)

        # Synthesise: find next occurrence of expected weekday
        target_wd = self._expected_weekday(entry)
        d = entry
        for _ in range(10):
            if d.weekday() == target_wd:
                return d
            d += timedelta(days=1)
        return None

    def get_dte(self, entry_date, expiry_date) -> int:
        """Return calendar days from entry_date to expiry_date (inclusive)."""
        e = _to_date(entry_date)
        x = _to_date(expiry_date)
        return max(0, (x - e).days)

    def get_entry_window(self, expiry_date) -> tuple[date, date]:
        """
        Return the (first_valid_entry, last_valid_entry) for a given expiry.
        We do not allow entry on expiry day or the day before expiry.
        Typically: first entry = Monday of that expiry week.
        """
        expiry = _to_date(expiry_date)
        last_entry   = expiry - timedelta(days=2)  # two days before expiry
        first_entry  = expiry - timedelta(days=6)  # Monday of same week (approx)
        # Skip weekends
        while first_entry.weekday() >= 5:
            first_entry += timedelta(days=1)
        return first_entry, last_entry

    def is_too_close_to_expiry(self, trade_date, expiry_date) -> bool:
        """True if trade_date is expiry day or the calendar day before expiry."""
        td = _to_date(trade_date)
        ex = _to_date(expiry_date)
        return (ex - td).days <= 1

    def get_all_expiries(self) -> list[date]:
        return list(self._expiries)

    def weekday_stats(self) -> dict:
        """Return count of expiries per weekday name."""
        stats: dict[str, int] = {}
        for e in self._expiries:
            name = _WD_NAMES[e.weekday()]
            stats[name] = stats.get(name, 0) + 1
        return stats

    def get_weekly_entry_dates(self, start_date, end_date) -> list[tuple[date, date]]:
        """
        Return list of (entry_date, expiry_date) pairs for all trading weeks
        in range. Entry date is the first trading day of each expiry week
        that is not within 2 days of expiry.
        """
        start = _to_date(start_date)
        end   = _to_date(end_date)
        pairs = []

        expiries_in_range = [e for e in self._expiries if start <= e <= end]
        # Also include expiries slightly outside range if entry falls inside
        extended = [e for e in self._expiries if e >= start - timedelta(days=7)]

        seen_expiries = set()
        for exp in extended:
            if exp in seen_expiries:
                continue
            first_entry, last_entry = self.get_entry_window(exp)
            # Entry is the first Monday of that expiry week within our range
            entry = first_entry
            while entry.weekday() >= 5:  # skip weekend
                entry += timedelta(days=1)
            if entry < start or entry > end:
                continue
            if entry > last_entry:
                continue
            pairs.append((entry, exp))
            seen_expiries.add(exp)

        return sorted(pairs)
