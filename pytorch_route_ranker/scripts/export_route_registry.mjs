import { mkdir, writeFile } from "node:fs/promises";

import { routeRegistry } from "../../src/data/routeRegistry.js";

const outputUrl = new URL("../data/route_registry.json", import.meta.url);

await mkdir(new URL("../data/", import.meta.url), { recursive: true });
await writeFile(outputUrl, `${JSON.stringify(routeRegistry, null, 2)}\n`, "utf8");

console.log(`Exported ${routeRegistry.length} approved routes to ${outputUrl.pathname}`);
