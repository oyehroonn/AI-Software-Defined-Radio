import { NextResponse } from "next/server";

export const runtime = "nodejs";

type RawStateVector = [
  string, // 0 - icao24
  string | null, // 1 - callsign
  string | null, // 2 - origin_country
  number | null, // 3 - time_position
  number | null, // 4 - last_contact
  number | null, // 5 - longitude
  number | null, // 6 - latitude
  number | null, // 7 - baro_altitude
  boolean | null, // 8 - on_ground
  number | null, // 9 - velocity
  number | null, // 10 - true_track
  number | null, // 11 - vertical_rate
  number[] | null, // 12 - sensors
  number | null, // 13 - geo_altitude
  string | null, // 14 - squawk
  boolean | null, // 15 - spi
  number | null // 16 - position_source
];

export async function GET() {
  try {
    const lamin = -85;
    const lamax = 85;
    const lomin = -180;
    const lomax = 180;

    const authUser = process.env.OPENSKY_USERNAME;
    const authPass = process.env.OPENSKY_PASSWORD;

    const url = new URL("https://opensky-network.org/api/states/all");
    url.searchParams.set("lamin", String(lamin));
    url.searchParams.set("lamax", String(lamax));
    url.searchParams.set("lomin", String(lomin));
    url.searchParams.set("lomax", String(lomax));

    const headers: HeadersInit = {};
    if (authUser && authPass) {
      const token = Buffer.from(`${authUser}:${authPass}`).toString("base64");
      headers.Authorization = `Basic ${token}`;
    }

    const res = await fetch(url.toString(), {
      headers,
      cache: "no-store",
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: "OpenSky request failed", status: res.status },
        { status: 502 },
      );
    }

    const json = (await res.json()) as {
      time: number;
      states: RawStateVector[] | null;
    };

    const time = json.time ?? Math.floor(Date.now() / 1000);
    const rawStates = json.states ?? [];

    let states = rawStates
      .map((s) => {
        const icao24 = s[0];
        const callsign = s[1]?.trim() || undefined;
        const lon = s[5] ?? undefined;
        const lat = s[6] ?? undefined;
        const baroAlt = s[7] ?? undefined;
        const velocityMs = s[9] ?? undefined;
        const headingDeg = s[10] ?? undefined;
        const geoAlt = s[13] ?? undefined;

        const altitudeMeters = geoAlt ?? baroAlt;

        return {
          icao24,
          callsign,
          lon,
          lat,
          altitudeMeters,
          headingDeg,
          velocityMs,
          lastUpdate: time,
        };
      })
      .filter(
        (s) =>
          typeof s.lat === "number" &&
          typeof s.lon === "number" &&
          s.lat! >= -90 &&
          s.lat! <= 90 &&
          s.lon! >= -180 &&
          s.lon! <= 180,
      );

    // Soft cap to around 1500 aircraft for performance
    const MAX_AIRCRAFT = 1500;
    if (states.length > MAX_AIRCRAFT) {
      // Prefer faster / higher aircraft for a more dynamic view
      states = states
        .sort(
          (a, b) =>
            (b.velocityMs ?? 0) + (b.altitudeMeters ?? 0) / 10 -
            ((a.velocityMs ?? 0) + (a.altitudeMeters ?? 0) / 10),
        )
        .slice(0, MAX_AIRCRAFT);
    }

    return NextResponse.json({ time, states });
  } catch (error) {
    console.error("OpenSky API error", error);
    return NextResponse.json(
      { error: "Unexpected error talking to OpenSky" },
      { status: 500 },
    );
  }
}

