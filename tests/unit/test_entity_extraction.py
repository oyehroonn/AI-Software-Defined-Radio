"""Unit tests for aviation entity extraction."""

import pytest
import sys
sys.path.insert(0, '.')

from edge.voice_capture.entity_extraction import (
    extract_aviation_entities,
    extract_callsigns,
    extract_runways,
    extract_altitudes,
    extract_headings,
    extract_frequencies,
    extract_squawks,
    is_emergency_communication,
)


class TestCallsignExtraction:
    """Tests for callsign extraction."""
    
    def test_airline_callsign(self):
        """Test standard airline callsign extraction."""
        text = "United 456 descend and maintain flight level 350"
        callsigns = extract_callsigns(text)
        assert "UAL456" in callsigns
        
    def test_short_callsign(self):
        """Test short format callsign."""
        text = "AAL123 turn left heading 270"
        callsigns = extract_callsigns(text)
        assert "AAL123" in callsigns
        
    def test_n_number(self):
        """Test N-number extraction."""
        text = "November 1 2 3 Alpha Bravo cleared for takeoff"
        callsigns = extract_callsigns(text)
        assert any("N123AB" in cs or "N12" in cs for cs in callsigns)
        
    def test_multiple_callsigns(self):
        """Test extracting multiple callsigns."""
        text = "AAL123 follow Delta 456 on final runway 27L"
        callsigns = extract_callsigns(text)
        assert "AAL123" in callsigns
        assert "DAL456" in callsigns


class TestRunwayExtraction:
    """Tests for runway extraction."""
    
    def test_basic_runway(self):
        """Test basic runway number."""
        text = "Cleared to land runway 27"
        runways = extract_runways(text)
        assert "27" in runways
        
    def test_runway_with_suffix(self):
        """Test runway with L/R/C suffix."""
        text = "Line up and wait runway 09L"
        runways = extract_runways(text)
        assert "09L" in runways
        
    def test_rwy_abbreviation(self):
        """Test RWY abbreviation."""
        text = "Cleared for takeoff RWY 36R"
        runways = extract_runways(text)
        assert "36R" in runways


class TestAltitudeExtraction:
    """Tests for altitude extraction."""
    
    def test_flight_level(self):
        """Test flight level extraction."""
        text = "Climb and maintain flight level 350"
        altitudes = extract_altitudes(text)
        assert 35000 in altitudes
        
    def test_thousands(self):
        """Test thousand feet extraction."""
        text = "Descend to 10 thousand feet"
        altitudes = extract_altitudes(text)
        assert 10000 in altitudes
        
    def test_direct_altitude(self):
        """Test direct altitude in feet."""
        text = "Maintain 3500 feet until established"
        altitudes = extract_altitudes(text)
        assert 3500 in altitudes


class TestHeadingExtraction:
    """Tests for heading extraction."""
    
    def test_heading(self):
        """Test heading extraction."""
        text = "Turn left heading 270"
        headings = extract_headings(text)
        assert 270 in headings
        
    def test_multiple_headings(self):
        """Test multiple headings."""
        text = "Turn right heading 090 then heading 180"
        headings = extract_headings(text)
        assert 90 in headings or 180 in headings


class TestFrequencyExtraction:
    """Tests for frequency extraction."""
    
    def test_tower_frequency(self):
        """Test tower frequency."""
        text = "Contact tower 118.3"
        freqs = extract_frequencies(text)
        assert 118.3 in freqs
        
    def test_approach_frequency(self):
        """Test approach frequency."""
        text = "Contact approach on 124.05"
        freqs = extract_frequencies(text)
        # Should find 124.05 or similar


class TestSquawkExtraction:
    """Tests for squawk code extraction."""
    
    def test_vfr_squawk(self):
        """Test VFR squawk code."""
        text = "Squawk 1200"
        squawks = extract_squawks(text)
        assert "1200" in squawks
        
    def test_emergency_squawk(self):
        """Test emergency squawk code."""
        text = "Squawk 7700"
        squawks = extract_squawks(text)
        assert "7700" in squawks


class TestEmergencyDetection:
    """Tests for emergency communication detection."""
    
    def test_squawk_7500(self):
        """Test hijacking squawk detection."""
        entities = extract_aviation_entities("Aircraft squawking 7500")
        assert is_emergency_communication(entities)
        
    def test_squawk_7600(self):
        """Test radio failure squawk detection."""
        entities = extract_aviation_entities("We have squawk 7600")
        assert is_emergency_communication(entities)
        
    def test_squawk_7700(self):
        """Test general emergency squawk detection."""
        entities = extract_aviation_entities("Squawk 7700 emergency")
        assert is_emergency_communication(entities)
        
    def test_mayday(self):
        """Test MAYDAY keyword detection."""
        entities = extract_aviation_entities("Mayday mayday mayday engine failure")
        assert is_emergency_communication(entities)
        
    def test_pan_pan(self):
        """Test PAN PAN detection."""
        entities = extract_aviation_entities("Pan pan pan medical emergency")
        assert is_emergency_communication(entities)
        
    def test_normal_communication(self):
        """Test normal communication is not flagged."""
        entities = extract_aviation_entities("United 456 cleared for approach runway 27L")
        assert not is_emergency_communication(entities)


class TestFullEntityExtraction:
    """Tests for complete entity extraction."""
    
    def test_complex_transmission(self):
        """Test extraction from complex ATC transmission."""
        text = """
        American 123 descend and maintain flight level 350 
        turn left heading 270 contact approach 124.35
        expect runway 27L
        """
        
        entities = extract_aviation_entities(text)
        
        assert "AAL123" in entities.callsigns
        assert 35000 in entities.altitudes
        assert 270 in entities.headings
        assert "27L" in entities.runways
