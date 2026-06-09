import { useEffect, useState } from "react";

const themeStorageKey = "amids-theme";

function getInitialTheme() {
  try {
    const savedTheme = window.localStorage.getItem(themeStorageKey);
    if (savedTheme === "dark" || savedTheme === "light") return savedTheme;
  } catch {
    // Continue with the operating-system preference when storage is unavailable.
  }

  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function AppHeader() {
  const [theme, setTheme] = useState(getInitialTheme);
  const isDarkTheme = theme === "dark";

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;

    try {
      window.localStorage.setItem(themeStorageKey, theme);
    } catch {
      // The selected theme still applies for this session when storage is unavailable.
    }
  }, [theme]);

  function toggleTheme() {
    setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"));
  }

  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">AMIDS Prototype</p>
        <h1>Aviation Role Control Panel</h1>
      </div>
      <button
        className="theme-toggle"
        type="button"
        aria-label={`Switch to ${isDarkTheme ? "light" : "dark"} theme`}
        aria-pressed={isDarkTheme}
        onClick={toggleTheme}
      >
        <span className="theme-toggle-track" aria-hidden="true">
          <span className="theme-toggle-thumb"></span>
        </span>
        {isDarkTheme ? "Light mode" : "Dark mode"}
      </button>
    </header>
  );
}
