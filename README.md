# temp
import os
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any

class USPSTokenError(Exception):
    pass

class USPSAddressClient:
    """
    Minimal USPS v3 client using a refresh token.
    """
    BASE = "https://apis.usps.com"
    TOKEN_URL = f"{BASE}/oauth2/v3/token"
    ADDRESS_URL = f"{BASE}/addresses/v3/address"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        timeout: float = 15.0,
        sleep_between: float = 0.10,
        max_retries: int = 2,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.timeout = timeout
        self.sleep_between = sleep_between
        self.max_retries = max_retries
        self._access_token = None
        self._access_token_expiry_epoch = 0  # (optional) not used here; we refresh on 401

    def _refresh_access_token(self) -> str:
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        r = requests.post(self.TOKEN_URL, json=payload, timeout=self.timeout)
        if not r.ok:
            raise USPSTokenError(f"USPS token refresh failed: {r.status_code} {r.text}")
        data = r.json()
        self._access_token = data.get("access_token")
        if not self._access_token:
            raise USPSTokenError("USPS token refresh response missing access_token")
        return self._access_token

    def _auth_header(self) -> Dict[str, str]:
        if not self._access_token:
            self._refresh_access_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def validate_address(
        self,
        street: Optional[str],
        secondary: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        zip5: Optional[str] = None,
        zip4: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Calls USPS /addresses/v3/address and returns parsed JSON (or None if not OK).
        """
        # Build query params with only non-empty values
        def _clean(s): 
            return (s or "").strip() or None

        params = {}
        if _clean(street):    params["streetAddress"] = _clean(street)
        if _clean(secondary): params["secondaryAddress"] = _clean(secondary)
        if _clean(city):      params["city"] = _clean(city)
        if _clean(state):     params["state"] = _clean(state)
        if _clean(zip5):      params["ZIPCode"] = _clean(zip5)
        if _clean(zip4):      params["ZIPPlus4"] = _clean(zip4)

        if not params.get("streetAddress"):
            return None  # need at least a primary street to get a useful result

        # Try request, refresh token on 401 once, simple backoff on 429
        retries = 0
        while retries <= self.max_retries:
            if retries > 0:
                time.sleep(0.5 * retries)
            hdrs = {"Accept": "application/json", **self._auth_header()}
            resp = requests.get(self.ADDRESS_URL, params=params, headers=hdrs, timeout=self.timeout)
            if resp.status_code == 401:
                # refresh and retry once
                self._refresh_access_token()
                retries += 1
                continue
            if resp.status_code == 429:
                time.sleep(1.0 + 0.5 * retries)
                retries += 1
                continue
            if resp.ok:
                time.sleep(self.sleep_between)
                return resp.json()
            # non-ok, non-401/429 -> break
            break

        return None

def dpv_code(usps_json: Dict[str, Any]) -> Optional[str]:
    return (usps_json or {}).get("additionalInfo", {}).get("DPVConfirmation")

def dpv_is_deliverable(usps_json: Dict[str, Any]) -> bool:
    """
    USPS DPV codes commonly seen:
      Y = deliverable (primary + secondary confirmed)
      D = primary confirmed, secondary missing
      S = primary confirmed, secondary present but unconfirmed
      N = not deliverable
    Treat Y/D/S as 'usable' for cleaning (primary is valid), N (or None) as invalid.
    """
    code = dpv_code(usps_json)
    return code in {"Y", "D", "S"}  # see DPV code references in notes

def apply_usps_to_row(row: pd.Series, usps_json: Dict[str, Any]) -> pd.Series:
    """
    Overwrite row fields using USPS standardized values.
    - address1 <- standardized streetAddress
    - address2 <- standardized secondaryAddress (or blank)
    - city/state/zip <- standardized city/state/ZIPCode (zip5 only)
    """
    addr = (usps_json or {}).get("address", {}) or {}
    street = addr.get("streetAddress") or ""
    secondary = addr.get("secondaryAddress") or ""
    city = addr.get("city") or row.get("city", "")
    state = addr.get("state") or row.get("state", "")
    zip5 = addr.get("ZIPCode") or row.get("zip", "")
    zip4 = addr.get("ZIPPlus4")

    # Overwrite core columns; create if missing
    row["address1"] = street
    row["address2"] = secondary

    # Update city/state/zip if the columns exist, else create them
    row["city"] = city
    row["state"] = state

    # Put 5-digit zip into 'zip' if present (most spreadsheets expect this)
    if "zip" in row.index or True:
        row["zip"] = str(zip5) if zip5 else ""

    # If you also keep a zip4 column, populate it
    if "zip4" in row.index:
        row["zip4"] = str(zip4) if zip4 else ""

    return row

def clean_dataframe_with_usps(
    df: pd.DataFrame,
    usps_client: USPSAddressClient,
    add_audit_cols: bool = False
) -> pd.DataFrame:
    """
    Modifies and returns df in place following the requested logic:
      1) Try address1 (+address2 as secondary). If USPS valid -> overwrite address1/2, city, state, zip.
      2) Else try address2 alone. If USPS valid -> put standardized into address1/2, update city/state/zip.
      3) Else uppercase address1 and address2.
    """

    required_cols = ["address1", "address2"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    audit_cols = ["usps_attempt", "usps_dpv", "usps_note"]
    if add_audit_cols:
        for c in audit_cols:
            if c not in df.columns:
                df[c] = ""

    def _process(row: pd.Series) -> pd.Series:
        a1 = (row.get("address1") or "").strip()
        a2 = (row.get("address2") or "").strip()
        city = (row.get("city") or "").strip()
        state = (row.get("state") or "").strip()
        zip5 = str(row.get("zip") or "").strip()
        zip4 = str(row.get("zip4") or "").strip() if "zip4" in row.index else None

        # Attempt 1: use address1 as street, address2 as secondary
        res1 = usps_client.validate_address(street=a1, secondary=a2, city=city, state=state, zip5=zip5, zip4=zip4)
        if res1 and dpv_is_deliverable(res1):
            row = apply_usps_to_row(row, res1)
            if add_audit_cols:
                row["usps_attempt"] = "address1"
                row["usps_dpv"] = dpv_code(res1) or ""
                row["usps_note"] = "validated by USPS using address1+address2"
            return row

        # Attempt 2: use address2 alone as street
        res2 = usps_client.validate_address(street=a2, secondary=None, city=city, state=state, zip5=zip5, zip4=zip4)
        if res2 and dpv_is_deliverable(res2):
            row = apply_usps_to_row(row, res2)
            if add_audit_cols:
                row["usps_attempt"] = "address2"
                row["usps_dpv"] = dpv_code(res2) or ""
                row["usps_note"] = "validated by USPS using address2 only"
            return row

        # Neither attempt validated -> uppercase original address1/address2
        row["address1"] = a1.upper()
        row["address2"] = a2.upper()
        if add_audit_cols:
            dpv = dpv_code(res1) or dpv_code(res2) or ""
            row["usps_attempt"] = "none"
            row["usps_dpv"] = dpv
            row["usps_note"] = "USPS could not validate; uppercased A1/A2"
        return row

    # Row-wise processing
    df = df.apply(_process, axis=1)
    return df


if __name__ == "__main__":
    # --- Example usage ---
    # Expect environment variables for secrets:
    #   USPS_CLIENT_ID, USPS_CLIENT_SECRET, USPS_REFRESH_TOKEN
    client = USPSAddressClient(
        client_id=os.environ.get("USPS_CLIENT_ID", "").strip(),
        client_secret=os.environ.get("USPS_CLIENT_SECRET", "").strip(),
        refresh_token=os.environ.get("USPS_REFRESH_TOKEN", "").strip(),
        sleep_between=0.10,   # USPS-friendly pacing for large jobs
    )

    # Load your CSV (must include at least address1/address2; city/state/zip optional but recommended)
    df = pd.read_csv("input_addresses.csv", dtype=str).fillna("")

    cleaned = clean_dataframe_with_usps(df, client, add_audit_cols=False)

    # Overwrite your original columns in-place has already happened; save if you want an output file:
    cleaned.to_csv("output_addresses_cleaned.csv", index=False)
    print("Done. Wrote output_addresses_cleaned.csv")
