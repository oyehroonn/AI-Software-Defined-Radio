"""LLM Copilot server for AeroSentry AI."""

import asyncio
import json
import logging
import os
from datetime import datetime, UTC
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="AeroSentry LLM Copilot")


class QueryRequest(BaseModel):
    """Natural language query request."""
    query: str
    context: Optional[dict] = None


class QueryResponse(BaseModel):
    """Query response."""
    answer: str
    sql_query: Optional[str] = None
    results: Optional[list] = None
    citations: list = []
    confidence: float = 1.0


class IncidentReportRequest(BaseModel):
    """Incident report generation request."""
    incident_id: str
    start_time: str
    end_time: str
    icao24_list: Optional[list[str]] = None


class IncidentReportResponse(BaseModel):
    """Generated incident report."""
    incident_id: str
    report: str
    summary: str
    recommendations: list[str]


# Lazy load components
_query_engine = None
_response_generator = None
_report_generator = None
_db_connection = None


async def get_db_connection():
    """Get or create database connection."""
    global _db_connection
    
    if _db_connection is None:
        try:
            import asyncpg
            _db_connection = await asyncpg.connect(
                host=os.getenv("TIMESCALE_HOST", "localhost"),
                port=int(os.getenv("TIMESCALE_PORT", "5432")),
                database=os.getenv("TIMESCALE_DB", "aerosentry"),
                user=os.getenv("TIMESCALE_USER", "aerosentry"),
                password=os.getenv("TIMESCALE_PASSWORD", "aerosentry_secure")
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            
    return _db_connection


def get_query_engine():
    """Get or create query engine."""
    global _query_engine
    
    if _query_engine is None:
        try:
            from cloud.llm_copilot.query_engine import QueryEngine
            _query_engine = QueryEngine()
        except Exception as e:
            logger.error(f"Failed to create query engine: {e}")
            
    return _query_engine


def get_response_generator():
    """Get or create response generator."""
    global _response_generator
    
    if _response_generator is None:
        try:
            from cloud.llm_copilot.response_generator import EvidenceGatedResponder
            
            llm_provider = os.getenv("LLM_PROVIDER", "openai")
            api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
            model = os.getenv("LLM_MODEL", "gpt-4")
            
            _response_generator = EvidenceGatedResponder(
                provider=llm_provider,
                api_key=api_key,
                model=model
            )
        except Exception as e:
            logger.error(f"Failed to create response generator: {e}")
            
    return _response_generator


def get_report_generator():
    """Get or create report generator."""
    global _report_generator
    
    if _report_generator is None:
        try:
            from cloud.llm_copilot.response_generator import IncidentReportGenerator
            
            llm_provider = os.getenv("LLM_PROVIDER", "openai")
            api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
            model = os.getenv("LLM_MODEL", "gpt-4")
            
            _report_generator = IncidentReportGenerator(
                provider=llm_provider,
                api_key=api_key,
                model=model
            )
        except Exception as e:
            logger.error(f"Failed to create report generator: {e}")
            
    return _report_generator


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db = await get_db_connection()
    
    return {
        "status": "healthy",
        "service": "llm-copilot",
        "database_connected": db is not None,
        "components": {
            "query_engine": _query_engine is not None,
            "response_generator": _response_generator is not None,
            "report_generator": _report_generator is not None
        }
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle natural language query."""
    query_engine = get_query_engine()
    response_gen = get_response_generator()
    db = await get_db_connection()
    
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not available")
        
    try:
        # Parse query to SQL
        sql_result = query_engine.parse_query(request.query)
        
        results = []
        if sql_result.get("sql") and db:
            # Execute query
            rows = await db.fetch(sql_result["sql"])
            results = [dict(r) for r in rows]
            
        # Generate response
        if response_gen:
            response = await response_gen.generate_response(
                query=request.query,
                results=results,
                context=request.context
            )
            
            return QueryResponse(
                answer=response.get("answer", "Unable to generate response"),
                sql_query=sql_result.get("sql"),
                results=results[:100],  # Limit results
                citations=response.get("citations", []),
                confidence=response.get("confidence", 1.0)
            )
        else:
            # Return raw results
            return QueryResponse(
                answer=f"Found {len(results)} results",
                sql_query=sql_result.get("sql"),
                results=results[:100],
                citations=[],
                confidence=1.0
            )
            
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report", response_model=IncidentReportResponse)
async def generate_report(request: IncidentReportRequest):
    """Generate incident report."""
    report_gen = get_report_generator()
    db = await get_db_connection()
    
    if not report_gen:
        raise HTTPException(status_code=503, detail="Report generator not available")
        
    try:
        # Parse times
        start_time = datetime.fromisoformat(request.start_time.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(request.end_time.replace("Z", "+00:00"))
        
        # Fetch data
        evidence = {}
        
        if db:
            # Alerts
            alert_query = """
                SELECT * FROM anomaly_alerts 
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp
            """
            alerts = await db.fetch(alert_query, start_time, end_time)
            evidence["alerts"] = [dict(a) for a in alerts]
            
            # Tracks
            if request.icao24_list:
                track_query = """
                    SELECT * FROM adsb_messages 
                    WHERE timestamp BETWEEN $1 AND $2
                    AND icao24 = ANY($3)
                    ORDER BY timestamp
                """
                tracks = await db.fetch(track_query, start_time, end_time, request.icao24_list)
            else:
                track_query = """
                    SELECT * FROM adsb_messages 
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp LIMIT 1000
                """
                tracks = await db.fetch(track_query, start_time, end_time)
                
            evidence["tracks"] = [dict(t) for t in tracks]
            
            # PHY features
            phy_query = """
                SELECT * FROM phy_features 
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp LIMIT 500
            """
            phy = await db.fetch(phy_query, start_time, end_time)
            evidence["phy_features"] = [dict(p) for p in phy]
            
        # Generate report
        report = await report_gen.generate_report(
            incident_id=request.incident_id,
            evidence=evidence
        )
        
        return IncidentReportResponse(
            incident_id=request.incident_id,
            report=report.get("full_report", ""),
            summary=report.get("summary", ""),
            recommendations=report.get("recommendations", [])
        )
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates")
async def list_query_templates():
    """List available query templates."""
    query_engine = get_query_engine()
    
    if not query_engine:
        return {"templates": []}
        
    return {"templates": list(query_engine.SQL_TEMPLATES.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
