import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const moduleName = process.argv[2];
const moduleArguments = process.argv.slice(3);

if (!moduleName) {
  console.error("Usage: node scripts/runPythonModule.mjs <python.module> [...arguments]");
  process.exit(1);
}

const configuredPython = process.env.AMIDS_PYTHON_COMMAND;
const projectVirtualEnvironmentPython =
  process.platform === "win32"
    ? resolve("pytorch_route_ranker", ".venv", "Scripts", "python.exe")
    : resolve("pytorch_route_ranker", ".venv", "bin", "python");
const candidates = configuredPython
  ? [{ command: configuredPython, prefixArguments: [] }]
  : process.platform === "win32"
    ? [
        ...(existsSync(projectVirtualEnvironmentPython)
          ? [{ command: projectVirtualEnvironmentPython, prefixArguments: [] }]
          : []),
        { command: "python", prefixArguments: [] },
        { command: "py", prefixArguments: ["-3.11"] },
        { command: "python3", prefixArguments: [] },
      ]
    : [
        ...(existsSync(projectVirtualEnvironmentPython)
          ? [{ command: projectVirtualEnvironmentPython, prefixArguments: [] }]
          : []),
        { command: "python3", prefixArguments: [] },
        { command: "python", prefixArguments: [] },
      ];

const selectedPython = candidates.find(({ command, prefixArguments }) => {
  const result = spawnSync(command, [...prefixArguments, "--version"], { stdio: "ignore" });
  return !result.error && result.status === 0;
});

if (!selectedPython) {
  console.error(
    "No usable Python installation was found. Activate the project virtual environment or set AMIDS_PYTHON_COMMAND."
  );
  process.exit(1);
}

const child = spawn(
  selectedPython.command,
  [...selectedPython.prefixArguments, "-m", moduleName, ...moduleArguments],
  {
    env: process.env,
    stdio: "inherit",
  }
);

child.on("error", (error) => {
  console.error(`Could not start Python: ${error.message}`);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exitCode = code ?? 1;
});
