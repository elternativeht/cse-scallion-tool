
from pathlib import Path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from typing import Any

from utilities import GOOGLE_PARSER_KEY_LIST, GOOGLE_PARSER_COL_LIST, GOOGLE_PARSER_REL_ROW_LIST

def google_api():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    creds = None
    if Path('token.json').exists() and Path('token.json').is_file(): #if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('sheets', 'v4', credentials=creds)
    sheet_api = service.spreadsheets()
    return sheet_api


def googlesheet_load(sheet_api: Any, sheet_id: str, sht_name: str, start_row: int):
    def single_range_parser(raw_result):
        return raw_result['values'][0][0]

    parsed_dict = {}

    assert start_row > 0

    row_list = [str(start_row + x) for x in GOOGLE_PARSER_REL_ROW_LIST]
    range_list = [sht_name + '!' + x + y for (x, y) in zip(GOOGLE_PARSER_COL_LIST, row_list)]

    range_values = sheet_api.values().batchGet(spreadsheetId=sheet_id,
                                               ranges=range_list,
                                               valueRenderOption='UNFORMATTED_VALUE',
                                               ).execute().get('valueRanges')

    range_values = list(map(single_range_parser, range_values))

    for key, range_value in zip(GOOGLE_PARSER_KEY_LIST, range_values):
        parsed_dict[key] = range_value

    '''
    0A - Overall
    0C - Exp Index
    1A - Month
    1B - Day
    5AB - Background per-cycle HC
    6M - Firing cycle num
    6O - Cold start mole
    15AC-AE cold start HC-CO-CO2 C
    16AC-AE 1P  HC-CO-CO2 C
    17AC-AE 2P  HC-CO-CO2 C
    18AC    3p  HC
    '''
    return parsed_dict

