"""Evidence-gated LLM response generation for RF Copilot."""

import logging
import re
import json
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LLM libraries
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are the RF Copilot for AeroSentry AI, an aviation RF monitoring and anomaly detection system.

Your role is to help users understand ADS-B data, anomaly alerts, and aviation RF activity. 

CRITICAL RULES FOR RESPONSES:
1. Every factual claim MUST be supported by evidence from the provided query results
2. Format evidence citations as [Evidence: field=value, source=table]
3. If evidence is insufficient to answer a question, say "Insufficient data to determine..."
4. NEVER speculate or make claims not supported by the data
5. When discussing anomalies, always reference the specific evidence that triggered the alert
6. Use aviation terminology correctly (ICAO24, callsign, squawk codes, etc.)
7. For safety-critical information, be especially careful to cite evidence

When analyzing alerts:
- Explain what the anomaly means in plain language
- Reference the specific rule triggers and scores
- Suggest possible explanations (but note uncertainty)
- Recommend appropriate follow-up actions

Available data sources:
- adsb_messages: Raw ADS-B position and velocity reports
- anomaly_alerts: Detected anomalies with severity and evidence
- track_windows: Aggregated track statistics
- phy_features: RF fingerprinting data
- voice_transcripts: ATC communication transcripts"""


RESPONSE_PROMPT = """Based on the following query and results, generate a helpful response.

User Query: {query}
SQL Executed: {sql}
Results ({row_count} rows): 
{results}

Generate a response that:
1. Directly answers the user's question
2. Cites specific evidence from the results
3. Explains any anomalies or notable findings
4. Suggests relevant follow-up queries if appropriate

Response:"""


@dataclass
class EvidenceCitation:
    """A citation linking a claim to evidence."""
    claim: str
    field: str
    value: Any
    source: str


@dataclass
class GatedResponse:
    """Response with evidence validation."""
    text: str
    citations: list[EvidenceCitation]
    confidence: float
    warnings: list[str]


class EvidenceGatedResponder:
    """Generate responses with mandatory evidence grounding."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        self.provider = llm_provider
        self.model = model
        self.client = None
        
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        elif llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None
            
    async def generate_response(
        self,
        query: str,
        sql: str,
        results: list[dict]
    ) -> GatedResponse:
        """
        Generate evidence-gated response.
        
        Args:
            query: Original natural language query
            sql: SQL that was executed
            results: Query results
            
        Returns:
            GatedResponse with validated citations
        """
        # Format results for prompt
        if len(results) > 20:
            results_str = json.dumps(results[:20], indent=2, default=str)
            results_str += f"\n... and {len(results) - 20} more rows"
        else:
            results_str = json.dumps(results, indent=2, default=str)
            
        prompt = RESPONSE_PROMPT.format(
            query=query,
            sql=sql,
            row_count=len(results),
            results=results_str
        )
        
        # Generate response
        if self.client:
            response_text = await self._call_llm(prompt)
        else:
            response_text = self._generate_fallback_response(query, results)
            
        # Validate and extract citations
        validated = self._validate_citations(response_text, results)
        
        return validated
        
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return f"Error generating response: {e}"
            
        return ""
        
    def _generate_fallback_response(
        self,
        query: str,
        results: list[dict]
    ) -> str:
        """Generate basic response without LLM."""
        if not results:
            return "No data found matching your query."
            
        query_lower = query.lower()
        
        if "aircraft" in query_lower:
            aircraft = set(r.get("icao24", "") for r in results if r.get("icao24"))
            return (f"Found {len(aircraft)} unique aircraft in the results. "
                   f"[Evidence: count={len(aircraft)}, source=adsb_messages]")
                   
        elif "alert" in query_lower:
            severities = {}
            for r in results:
                sev = r.get("severity", "unknown")
                severities[sev] = severities.get(sev, 0) + 1
            summary = ", ".join(f"{k}: {v}" for k, v in severities.items())
            return (f"Found {len(results)} alerts. Breakdown by severity: {summary}. "
                   f"[Evidence: total={len(results)}, source=anomaly_alerts]")
                   
        elif "emergency" in query_lower or "squawk" in query_lower:
            emergencies = [r for r in results if r.get("squawk") in ["7500", "7600", "7700"]]
            if emergencies:
                return (f"Found {len(emergencies)} emergency squawk codes. "
                       f"[Evidence: count={len(emergencies)}, source=adsb_messages]")
            return "No emergency squawk codes found in the specified time range."
            
        return f"Query returned {len(results)} results."
        
    def _validate_citations(
        self,
        response_text: str,
        results: list[dict]
    ) -> GatedResponse:
        """Validate all citations in response against actual data."""
        citations = []
        warnings = []
        
        # Extract citation patterns
        citation_pattern = r'\[Evidence:\s*(\w+)=([^,\]]+)(?:,\s*source=(\w+))?\]'
        matches = re.findall(citation_pattern, response_text)
        
        for field, value, source in matches:
            # Verify citation exists in results
            found = False
            for row in results:
                if field in row:
                    row_value = str(row[field])
                    if row_value == value or value in row_value:
                        found = True
                        citations.append(EvidenceCitation(
                            claim="",
                            field=field,
                            value=value,
                            source=source or "query_results"
                        ))
                        break
                        
            if not found:
                warnings.append(f"Citation not verified: {field}={value}")
                
        # Calculate confidence based on citation verification
        if matches:
            verified_ratio = len(citations) / len(matches)
        else:
            verified_ratio = 0.5  # No citations = moderate confidence
            
        confidence = min(1.0, verified_ratio * 0.9 + 0.1)
        
        return GatedResponse(
            text=response_text,
            citations=citations,
            confidence=confidence,
            warnings=warnings
        )


class IncidentReportGenerator:
    """Generate incident reports from anomaly data."""
    
    def __init__(self, responder: Optional[EvidenceGatedResponder] = None):
        self.responder = responder
        
    async def generate_report(
        self,
        alert_data: dict,
        track_history: list[dict],
        phy_evidence: Optional[list[dict]] = None
    ) -> str:
        """Generate formatted incident report."""
        report_sections = []
        
        # Header
        report_sections.append(f"""# Incident Report: {alert_data.get('alert_id', 'Unknown')}

**Generated:** {alert_data.get('timestamp', 'N/A')}
**Aircraft:** {alert_data.get('icao24', 'Unknown')} ({alert_data.get('callsign', 'No callsign')})
**Severity:** {alert_data.get('severity', 'Unknown').upper()}
**Alert Type:** {alert_data.get('alert_type', 'Unknown')}

---
""")
        
        # Alert Details
        report_sections.append("""## Alert Details

""")
        
        if alert_data.get('anomaly_score'):
            report_sections.append(f"**Anomaly Score:** {alert_data['anomaly_score']:.3f}\n\n")
            
        # Rule triggers
        triggers = alert_data.get('rule_triggers', [])
        if triggers:
            report_sections.append("### Triggered Rules\n\n")
            for trigger in triggers:
                if isinstance(trigger, dict):
                    report_sections.append(
                        f"- **{trigger.get('rule_id', 'Unknown')}** "
                        f"({trigger.get('severity', 'N/A')}): "
                        f"{trigger.get('description', 'No description')}\n"
                    )
                    
        # Evidence
        evidence = alert_data.get('evidence', {})
        if evidence:
            report_sections.append("\n### Evidence\n\n")
            report_sections.append("| Metric | Value |\n|--------|-------|\n")
            for key, value in evidence.items():
                if isinstance(value, float):
                    report_sections.append(f"| {key} | {value:.4f} |\n")
                else:
                    report_sections.append(f"| {key} | {value} |\n")
                    
        # Track History Summary
        if track_history:
            report_sections.append(f"\n## Track History ({len(track_history)} points)\n\n")
            
            # Position summary
            lats = [t['latitude'] for t in track_history if t.get('latitude')]
            lons = [t['longitude'] for t in track_history if t.get('longitude')]
            alts = [t['altitude'] for t in track_history if t.get('altitude')]
            
            if lats and lons:
                report_sections.append(
                    f"**Position Range:** "
                    f"Lat [{min(lats):.4f}, {max(lats):.4f}], "
                    f"Lon [{min(lons):.4f}, {max(lons):.4f}]\n\n"
                )
            if alts:
                report_sections.append(
                    f"**Altitude Range:** {min(alts)} - {max(alts)} ft\n\n"
                )
                
        # PHY Evidence
        if phy_evidence:
            report_sections.append("\n## PHY-Layer Analysis\n\n")
            report_sections.append(f"**Samples Analyzed:** {len(phy_evidence)}\n\n")
            
        # Recommendations
        report_sections.append("""
## Recommendations

Based on the alert type and evidence:

""")
        
        alert_type = alert_data.get('alert_type', '')
        severity = alert_data.get('severity', '')
        
        if severity == 'critical':
            report_sections.append("1. **IMMEDIATE ACTION REQUIRED** - Review and verify with additional sources\n")
        if 'spoof' in alert_type.lower():
            report_sections.append("2. Cross-reference with other ADS-B receivers in the area\n")
            report_sections.append("3. Check for PHY-layer consistency anomalies\n")
        if 'kinematic' in alert_type.lower():
            report_sections.append("2. Review track history for impossible maneuvers\n")
            report_sections.append("3. Consider sensor malfunction as alternative explanation\n")
            
        report_sections.append("\n---\n*Report generated by AeroSentry AI*\n")
        
        return "".join(report_sections)
