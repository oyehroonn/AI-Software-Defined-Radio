"""Aviation entity extraction from ATC transcripts."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AviationEntities:
    """Extracted aviation entities from transcript."""
    callsigns: list[str]
    runways: list[str]
    altitudes: list[int]
    headings: list[int]
    frequencies: list[float]
    instructions: list[str]
    squawks: list[str]
    speeds: list[int]
    waypoints: list[str]
    raw_text: str


# Phonetic alphabet mapping
PHONETIC_ALPHABET = {
    "ALPHA": "A", "BRAVO": "B", "CHARLIE": "C", "DELTA": "D",
    "ECHO": "E", "FOXTROT": "F", "GOLF": "G", "HOTEL": "H",
    "INDIA": "I", "JULIET": "J", "KILO": "K", "LIMA": "L",
    "MIKE": "M", "NOVEMBER": "N", "OSCAR": "O", "PAPA": "P",
    "QUEBEC": "Q", "ROMEO": "R", "SIERRA": "S", "TANGO": "T",
    "UNIFORM": "U", "VICTOR": "V", "WHISKEY": "W", "XRAY": "X",
    "YANKEE": "Y", "ZULU": "Z"
}

# Number words
NUMBER_WORDS = {
    "ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
    "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9",
    "NINER": "9"  # Aviation pronunciation
}

# Common airline callsign prefixes
AIRLINE_CALLSIGNS = {
    "AMERICAN": "AAL", "UNITED": "UAL", "DELTA": "DAL",
    "SOUTHWEST": "SWA", "JETBLUE": "JBU", "ALASKA": "ASA",
    "SPIRIT": "NKS", "FRONTIER": "FFT", "HAWAIIAN": "HAL",
    "CACTUS": "AWE", "SPEEDBIRD": "BAW", "LUFTHANSA": "DLH",
    "AIR FRANCE": "AFR", "KLMROYAL": "KLM", "SHAMROCK": "EIN"
}

# Regex patterns
PATTERNS = {
    # Standard airline callsigns: AAL123, UAL456A
    "airline_callsign": r"\b([A-Z]{2,3})\s*(\d{1,4}[A-Z]?)\b",
    
    # N-numbers: N123AB, NOVEMBER 1 2 3 ALPHA BRAVO
    "n_number": r"\b(N|NOVEMBER)\s*(\d{1,5}[A-Z]{0,2})\b",
    
    # Runways: RUNWAY 27L, RWY 09R
    "runway": r"\b(?:RUNWAY|RWY)\s*(\d{1,2})\s*([LRC]?)\b",
    
    # Altitudes: FLIGHT LEVEL 350, 10000 FEET, CLIMB TO 8000
    "flight_level": r"\bFLIGHT\s*LEVEL\s*(\d{2,3})\b",
    "altitude_feet": r"\b(\d{1,2})\s*THOUSAND(?:\s+FEET)?\b",
    "altitude_direct": r"\b(\d{3,5})\s*(?:FEET|FT)\b",
    
    # Headings: HEADING 270, TURN LEFT HEADING 180
    "heading": r"\bHEADING\s*(\d{3})\b",
    
    # Frequencies: CONTACT TOWER 118.3, FREQUENCY 124.0
    "frequency": r"\b(\d{3})\s*[.]?\s*(\d{1,3})\b",
    
    # Squawk: SQUAWK 1200, SQUAWK 7700
    "squawk": r"\bSQUAWK\s*(\d{4})\b",
    
    # Speed: SPEED 250, MAINTAIN 180 KNOTS
    "speed": r"\b(\d{2,3})\s*KNOTS\b",
    
    # Instructions
    "cleared": r"\bCLEARED\s+(?:FOR\s+)?(\w+(?:\s+\w+)?)",
    "maintain": r"\bMAINTAIN\s+(\d+)",
    "descend": r"\bDESCEND\s+(?:AND\s+MAINTAIN\s+)?(\d+)",
    "climb": r"\bCLIMB\s+(?:AND\s+MAINTAIN\s+)?(\d+)",
    "turn": r"\bTURN\s+(LEFT|RIGHT)",
    "contact": r"\bCONTACT\s+(\w+)",
}


def normalize_text(text: str) -> str:
    """Normalize text for pattern matching."""
    text = text.upper()
    
    # Replace phonetic alphabet with letters
    for phonetic, letter in PHONETIC_ALPHABET.items():
        text = re.sub(rf"\b{phonetic}\b", letter, text)
        
    # Replace number words with digits
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
        
    return text


def extract_callsigns(text: str) -> list[str]:
    """Extract aircraft callsigns from text."""
    callsigns = []
    normalized = normalize_text(text)
    
    # Standard airline callsigns
    for match in re.finditer(PATTERNS["airline_callsign"], normalized):
        prefix, number = match.groups()
        callsigns.append(f"{prefix}{number}")
        
    # N-numbers
    for match in re.finditer(PATTERNS["n_number"], normalized):
        prefix, number = match.groups()
        if prefix == "NOVEMBER":
            prefix = "N"
        callsigns.append(f"{prefix}{number}")
        
    # Airline name callsigns (e.g., "AMERICAN 123")
    text_upper = text.upper()
    for airline, code in AIRLINE_CALLSIGNS.items():
        pattern = rf"\b{airline}\s*(\d{{1,4}}[A-Z]?)\b"
        for match in re.finditer(pattern, text_upper):
            callsigns.append(f"{code}{match.group(1)}")
            
    return list(set(callsigns))


def extract_runways(text: str) -> list[str]:
    """Extract runway designations from text."""
    runways = []
    normalized = normalize_text(text)
    
    for match in re.finditer(PATTERNS["runway"], normalized):
        number, suffix = match.groups()
        runway = f"{int(number):02d}{suffix}"
        runways.append(runway)
        
    return list(set(runways))


def extract_altitudes(text: str) -> list[int]:
    """Extract altitudes from text."""
    altitudes = []
    normalized = normalize_text(text)
    
    # Flight levels (FL350 = 35000 ft)
    for match in re.finditer(PATTERNS["flight_level"], normalized):
        fl = int(match.group(1))
        altitudes.append(fl * 100)
        
    # X thousand feet
    for match in re.finditer(PATTERNS["altitude_feet"], normalized):
        thousands = int(match.group(1))
        altitudes.append(thousands * 1000)
        
    # Direct altitude in feet
    for match in re.finditer(PATTERNS["altitude_direct"], normalized):
        altitudes.append(int(match.group(1)))
        
    return list(set(altitudes))


def extract_headings(text: str) -> list[int]:
    """Extract headings from text."""
    headings = []
    normalized = normalize_text(text)
    
    for match in re.finditer(PATTERNS["heading"], normalized):
        heading = int(match.group(1))
        if 0 <= heading <= 360:
            headings.append(heading)
            
    return list(set(headings))


def extract_frequencies(text: str) -> list[float]:
    """Extract radio frequencies from text."""
    frequencies = []
    normalized = normalize_text(text)
    
    for match in re.finditer(PATTERNS["frequency"], normalized):
        whole, decimal = match.groups()
        freq = float(f"{whole}.{decimal}")
        # VHF airband is 118-137 MHz
        if 118 <= freq <= 137:
            frequencies.append(freq)
            
    return list(set(frequencies))


def extract_squawks(text: str) -> list[str]:
    """Extract squawk codes from text."""
    squawks = []
    normalized = normalize_text(text)
    
    for match in re.finditer(PATTERNS["squawk"], normalized):
        squawk = match.group(1)
        if len(squawk) == 4 and all(c in "01234567" for c in squawk):
            squawks.append(squawk)
            
    return list(set(squawks))


def extract_speeds(text: str) -> list[int]:
    """Extract speeds from text."""
    speeds = []
    normalized = normalize_text(text)
    
    for match in re.finditer(PATTERNS["speed"], normalized):
        speed = int(match.group(1))
        if 50 <= speed <= 600:  # Reasonable aircraft speeds
            speeds.append(speed)
            
    return list(set(speeds))


def extract_instructions(text: str) -> list[str]:
    """Extract ATC instructions from text."""
    instructions = []
    text_upper = text.upper()
    
    # Cleared for...
    for match in re.finditer(PATTERNS["cleared"], text_upper):
        instructions.append(f"CLEARED {match.group(1)}")
        
    # Turn left/right
    for match in re.finditer(PATTERNS["turn"], text_upper):
        instructions.append(f"TURN {match.group(1)}")
        
    # Contact...
    for match in re.finditer(PATTERNS["contact"], text_upper):
        instructions.append(f"CONTACT {match.group(1)}")
        
    # Climb/descend
    if re.search(r"\bCLIMB\b", text_upper):
        instructions.append("CLIMB")
    if re.search(r"\bDESCEND\b", text_upper):
        instructions.append("DESCEND")
    if re.search(r"\bHOLD\b", text_upper):
        instructions.append("HOLD")
    if re.search(r"\bEXPEDITE\b", text_upper):
        instructions.append("EXPEDITE")
        
    return instructions


def extract_waypoints(text: str) -> list[str]:
    """Extract waypoint/fix names from text."""
    waypoints = []
    normalized = normalize_text(text)
    
    # 5-letter waypoint names (ICAO standard)
    waypoint_pattern = r"\b([A-Z]{5})\b"
    
    # Common words to exclude
    exclude = {"ROGER", "CLEAR", "TOWER", "RADAR", "DELTA", "ALPHA", 
               "BRAVO", "OSCAR", "INDIA", "ROMEO", "HOTEL", "NORTH",
               "SOUTH", "SPEED", "LEVEL", "HEAVY", "LIGHT"}
    
    for match in re.finditer(waypoint_pattern, normalized):
        waypoint = match.group(1)
        if waypoint not in exclude:
            waypoints.append(waypoint)
            
    return list(set(waypoints))


def extract_aviation_entities(transcript: str) -> AviationEntities:
    """
    Extract all aviation entities from a transcript.
    
    Args:
        transcript: Raw transcript text
        
    Returns:
        AviationEntities with extracted information
    """
    return AviationEntities(
        callsigns=extract_callsigns(transcript),
        runways=extract_runways(transcript),
        altitudes=extract_altitudes(transcript),
        headings=extract_headings(transcript),
        frequencies=extract_frequencies(transcript),
        instructions=extract_instructions(transcript),
        squawks=extract_squawks(transcript),
        speeds=extract_speeds(transcript),
        waypoints=extract_waypoints(transcript),
        raw_text=transcript
    )


def is_emergency_communication(entities: AviationEntities) -> bool:
    """Check if communication indicates an emergency."""
    emergency_squawks = {"7500", "7600", "7700"}
    
    if any(s in emergency_squawks for s in entities.squawks):
        return True
        
    emergency_keywords = ["MAYDAY", "PAN PAN", "EMERGENCY", "DECLARING"]
    text_upper = entities.raw_text.upper()
    
    return any(kw in text_upper for kw in emergency_keywords)
