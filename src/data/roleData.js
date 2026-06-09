const amidsSimulationHref = "/amids-simulation.html";

function toId(label) {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function createNavigationLinks(items) {
  return items.map((item, index) => {
    const id = item.id ?? toId(item.label);

    return {
      ...item,
      id,
      // Replace this simulated URL with the relevant confidential AMIDS URL,
      // dashboard route, document hyperlink, or internal tool endpoint.
      href: item.href ?? `${amidsSimulationHref}?resource=${encodeURIComponent(id)}`,
      priority: item.priority ?? (index === 0 ? "primary" : "secondary"),
      keywords: item.keywords ?? [],
    };
  });
}

export const roleData = {
  pilot: {
    title: "Pilot",
    shortLabel: "Flight deck decisions",
    panelTitle: "Pilot Weather Navigation",
    description:
      "Prioritizes flight safety, approach planning, fuel impact, diversion criteria, and cockpit-ready weather interpretation.",
    metrics: [
      ["Wind", "270/18G28"],
      ["Visibility", "6 km"],
      ["QNH", "1008 hPa"],
    ],
    categories: {
      wind: {
        summary: "Runway crosswind, tailwind, turbulence, and gust checks.",
        detail:
          "Use these links when a pilot needs wind information for departure, approach, runway selection, or alternate planning.",
        links: createNavigationLinks([
          {
            label: "Runway wind component",
            description: "Link to the AMIDS page that calculates crosswind and tailwind by runway.",
          },
          {
            label: "Wind shear and gust alerts",
            description: "Link to current wind shear warnings, gust trends, and relevant alert products.",
          },
          {
            label: "Alternate aerodrome winds",
            description: "Link to wind conditions at nominated alternates and nearby diversion options.",
          },
        ]),
      },
      precipitation: {
        summary: "Rain, snow, hail, freezing rain, and storm cell impact.",
        detail:
          "Use these links when precipitation could affect braking action, route selection, fuel, or diversion decisions.",
        links: createNavigationLinks([
          {
            label: "Radar and precipitation layer",
            description: "Link to the AMIDS radar or precipitation display for the route and destination.",
          },
          {
            label: "Runway condition impact",
            description: "Link to runway contamination, braking action, or airport condition reports.",
          },
          {
            label: "Fuel and diversion impact",
            description: "Link to the tool or report used to assess fuel penalties and diversion risk.",
          },
          {
            label: "Convective weather products",
            description: "Link to thunderstorm, SIGMET, or convective forecast products.",
          },
        ]),
      },
      visibility: {
        summary: "METAR visibility, RVR, fog, haze, and minima checks.",
        detail:
          "Use these links when a pilot needs to compare visibility against departure, approach, or alternate minima.",
        links: createNavigationLinks([
          {
            label: "Approach minima comparison",
            description: "Link to the page that compares current visibility or RVR with approach minima.",
          },
          {
            label: "RVR and visibility trend",
            description: "Link to METAR, SPECI, RVR trend, fog, haze, and low visibility observations.",
          },
          {
            label: "Alternate visibility status",
            description: "Link to visibility and forecast status for valid alternate aerodromes.",
          },
        ]),
      },
      temperature: {
        summary: "Icing, density altitude, performance, and cold weather limits.",
        detail:
          "Use these links when temperature affects aircraft performance, icing, anti-ice use, or cold weather operations.",
        links: createNavigationLinks([
          {
            label: "Density altitude and performance",
            description: "Link to performance-sensitive temperature and density altitude information.",
          },
          {
            label: "Icing level and freezing risk",
            description: "Link to icing forecasts, freezing level, and temperature/dew point spread.",
          },
          {
            label: "Cold weather limits",
            description: "Link to cold weather operational limits, anti-ice guidance, or holdover context.",
          },
        ]),
      },
      altimeter: {
        summary: "QNH, pressure trend, transition level, and altimetry risk.",
        detail:
          "Use these links when pressure settings or rapid pressure changes could affect altitude awareness.",
        links: createNavigationLinks([
          {
            label: "Current QNH and pressure trend",
            description: "Link to current QNH, pressure tendency, and recent METAR pressure values.",
          },
          {
            label: "Transition level reference",
            description: "Link to transition altitude or transition level guidance for the airport or FIR.",
          },
        ]),
      },
      ash: {
        summary: "Volcanic ash advisories, no-fly zones, and route risk.",
        detail:
          "Use these links when volcanic ash could affect routing, engine risk, alternates, or airspace restrictions.",
        warning: true,
        links: createNavigationLinks([
          {
            label: "VAAC advisory",
            description: "Link to the active volcanic ash advisory product or AMIDS VAAC summary.",
            priority: "warning",
          },
          {
            label: "Ash cloud route overlay",
            description: "Link to ash polygons, altitude bands, and route intersection tools.",
          },
          {
            label: "Clean alternate options",
            description: "Link to alternates away from contaminated airspace.",
          },
        ]),
      },
    },
  },
  atc: {
    title: "ATC",
    shortLabel: "Traffic separation",
    panelTitle: "ATC Weather Navigation",
    description:
      "Focuses on traffic flow, runway configuration, separation standards, sector capacity, and timely pilot advisories.",
    metrics: [
      ["Active RWY", "25L/25R"],
      ["Ceiling", "1,200 ft"],
      ["Flow", "Moderate"],
    ],
    categories: {
      wind: {
        summary: "Runway selection, wind shear broadcasts, and spacing impact.",
        detail:
          "Use these links when controllers need wind information for runway configuration, spacing, alerts, or ATIS updates.",
        links: createNavigationLinks([
          {
            label: "Runway configuration support",
            description: "Link to the AMIDS page that supports runway choice based on current and forecast wind.",
          },
          {
            label: "Wind shear broadcast",
            description: "Link to wind shear warnings, broadcast wording, or controller alert status.",
          },
          {
            label: "ATIS wind update",
            description: "Link to the workflow or page used to update ATIS wind information.",
          },
        ]),
      },
      precipitation: {
        summary: "Storm cells, runway wetness, braking action, and delays.",
        detail:
          "Use these links when precipitation affects traffic flow, runway occupancy, ground movement, or delay advisories.",
        links: createNavigationLinks([
          {
            label: "Traffic and radar overlay",
            description: "Link to the combined traffic and weather radar display.",
          },
          {
            label: "Runway inspection trigger",
            description: "Link to runway wetness, contamination, or inspection coordination information.",
          },
          {
            label: "Delay advisory support",
            description: "Link to the operational page used to issue or review delay advisories.",
          },
        ]),
      },
      visibility: {
        summary: "Low visibility procedures, RVR, and separation changes.",
        detail:
          "Use these links when visibility or RVR affects airport operating category, procedures, or spacing.",
        links: createNavigationLinks([
          {
            label: "Low visibility procedure checklist",
            description: "Link to LVP activation status, criteria, or checklist material.",
          },
          {
            label: "RVR broadcast values",
            description: "Link to current RVR and visibility values used for controller broadcasts.",
          },
          {
            label: "Flow rate adjustment",
            description: "Link to the tool or guidance for arrival/departure spacing changes.",
          },
        ]),
      },
      temperature: {
        summary: "Icing reports, deicing demand, and runway contamination risk.",
        detail:
          "Use these links when temperature affects deicing coordination, pilot reports, or runway surface risk.",
        links: createNavigationLinks([
          {
            label: "PIREP and icing reports",
            description: "Link to pilot reports, icing alerts, and sector-relevant weather reports.",
          },
          {
            label: "Deicing queue status",
            description: "Link to ground coordination, deicing queue, or surface movement information.",
          },
        ]),
      },
      altimeter: {
        summary: "QNH distribution, pressure changes, and readback attention.",
        detail:
          "Use these links when QNH must be synchronized across displays, broadcasts, and controller readbacks.",
        links: createNavigationLinks([
          {
            label: "ATIS QNH update",
            description: "Link to the ATIS or broadcast workflow for QNH updates.",
          },
          {
            label: "Pressure change alert",
            description: "Link to pressure trend alerts and rapid pressure change monitoring.",
          },
          {
            label: "Readback verification",
            description: "Link to readback guidance, controller prompts, or relevant safety notices.",
          },
        ]),
      },
      ash: {
        summary: "Airspace restrictions, reroutes, and emergency coordination.",
        detail:
          "Use these links when volcanic ash affects sectors, reroutes, emergency broadcasts, or airspace restrictions.",
        warning: true,
        links: createNavigationLinks([
          {
            label: "Sector restriction status",
            description: "Link to current sector closures or restricted airspace information.",
            priority: "warning",
          },
          {
            label: "Reroute coordination",
            description: "Link to flow management, reroute, or coordination tools.",
          },
          {
            label: "Ash advisory broadcast",
            description: "Link to controller broadcast wording or active volcanic ash advisory information.",
          },
        ]),
      },
    },
  },
  dispatcher: {
    title: "Dispatcher",
    shortLabel: "Route planning",
    panelTitle: "Dispatcher Weather Navigation",
    description:
      "Balances flight release, fuel strategy, alternates, reroutes, schedule impact, and operational risk across multiple flights.",
    metrics: [
      ["Risk", "Elevated"],
      ["Alternates", "2 valid"],
      ["ETD", "18:50 HKT"],
    ],
    categories: {
      wind: {
        summary: "Fuel burn, route winds, alternates, and runway acceptance.",
        detail:
          "Use these links when dispatch needs wind information for release planning, fuel, alternates, or runway limits.",
        links: createNavigationLinks([
          {
            label: "Fuel impact model",
            description: "Link to the AMIDS tool that models route winds and fuel penalties.",
          },
          {
            label: "Alternate comparison",
            description: "Link to alternate aerodrome wind and operational suitability information.",
          },
          {
            label: "Runway limit check",
            description: "Link to runway acceptance, tailwind limits, or operational limit references.",
          },
        ]),
      },
      precipitation: {
        summary: "Convective reroutes, delay programs, and airport disruption.",
        detail:
          "Use these links when precipitation could drive reroutes, delays, deicing, or disruption across the network.",
        links: createNavigationLinks([
          {
            label: "Reroute option builder",
            description: "Link to the route planning tool for avoiding precipitation or convective weather.",
          },
          {
            label: "Delay cost estimate",
            description: "Link to schedule impact, delay program, or operational cost information.",
          },
          {
            label: "Operations notification",
            description: "Link to the workflow for notifying network control or operations teams.",
          },
        ]),
      },
      visibility: {
        summary: "Dispatch minima, alternate minima, and arrival acceptance.",
        detail:
          "Use these links when visibility affects legal release, destination acceptance, or alternate validity.",
        links: createNavigationLinks([
          {
            label: "Release validation",
            description: "Link to dispatch release validation against current and forecast visibility.",
          },
          {
            label: "Alternate minima check",
            description: "Link to alternate minima and forecast visibility comparison.",
          },
          {
            label: "Forecast trend monitor",
            description: "Link to visibility forecast trend, TAF, METAR, or SPECI monitoring.",
          },
        ]),
      },
      temperature: {
        summary: "Payload limits, icing, deicing time, and performance planning.",
        detail:
          "Use these links when temperature affects payload, holdover time, route icing, or flight release constraints.",
        links: createNavigationLinks([
          {
            label: "Payload penalty estimate",
            description: "Link to performance or payload calculations affected by temperature.",
          },
          {
            label: "Holdover time guidance",
            description: "Link to deicing holdover guidance, fluid tables, or airport deicing information.",
          },
          {
            label: "Icing route review",
            description: "Link to icing forecasts and route-level icing risk products.",
          },
        ]),
      },
      altimeter: {
        summary: "QNH trend, airport pressure risk, and operational briefings.",
        detail:
          "Use these links when pressure information should be added to releases, briefings, or operational alerts.",
        links: createNavigationLinks([
          {
            label: "Dispatch release note",
            description: "Link to the release note or briefing workflow for QNH-related concerns.",
          },
          {
            label: "QNH trend monitor",
            description: "Link to airport pressure trend data and pressure drop alerts.",
          },
        ]),
      },
      ash: {
        summary: "Route closure, engine risk, alternates, and fleet disruption.",
        detail:
          "Use these links when volcanic ash affects routes, flight levels, alternates, maintenance exposure, or fleet recovery.",
        warning: true,
        links: createNavigationLinks([
          {
            label: "Exposed flights list",
            description: "Link to the AMIDS view that identifies flights intersecting ash-affected areas.",
            priority: "warning",
          },
          {
            label: "Ash-safe reroute generator",
            description: "Link to reroute generation or flight planning tools.",
          },
          {
            label: "Network control alert",
            description: "Link to the workflow for alerting network control or disruption management teams.",
          },
          {
            label: "Maintenance exposure review",
            description: "Link to engine exposure, maintenance follow-up, or aircraft status information.",
          },
        ]),
      },
    },
  },
};
