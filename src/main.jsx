import React, { useEffect, useMemo, useState } from "react"; // Imports React and the hooks used for state, effects, and memoized values.
import { createRoot } from "react-dom/client"; // Imports the React DOM function that mounts the app into index.html.
import "./styles.css";
import { roleData } from "./data/roleData"; 
import { categoryLabels } from "./data/categoryLabels"; 
import AppHeader from "./components/AppHeader"; 
import RolePicker from "./components/RolePicker"; 
import ControlPanel from "./components/ControlPanel"; 
import AssistantPanel from "./components/AssistantPanel"; 
import RuleManagerPanel from "./components/RuleManagerPanel"; 
import { defaultPreferences } from "./data/assistantConfig"; 
import { loadUserProfile } from "./services/assistantEngine"; 

const transitionDuration = 150; // Sets the animation delay used when switching screens.
const userId = "prototype-user"; // Identifies the current user for personalization (to be changed when user accounts exist)

function unique(values) { // Helper that removes empty values and duplicates.
  return [...new Set(values.filter(Boolean))]; // Filters falsy values, turn into a Set to remove duplicates, then converts it back to an array.
}

function mergePreferencesWithProfile(preferences, profile) { // Combines default preferences with learned profile data.
  if (!profile) return preferences; // Uses defaults unchanged when no profile has been loaded.

  return { // Returns a new preferences object instead of mutating the original one.
    ...preferences, // Copies all existing preference properties.
    preferredCategories: unique([ // Creates a clean combined category preference list.
      ...preferences.preferredCategories, // Adds the default preferred categories.
      ...(profile.frequentTopics ?? []), // Adds learned frequent topics, or an empty array if missing.
    ]),
    preferredLinks: unique([ // Creates a clean combined preferred route/link list.
      ...preferences.preferredLinks, // Adds the default preferred links.
      ...(profile.preferredRouteIds ?? []), // Adds learned preferred route IDs, or an empty array if missing.
    ]),
  };
}

function findRoleLinkById(role, linkId) { // Finds a manual roleData link by its ID.
  return Object.values(role.categories) // Gets every category object for the selected role.
    .flatMap((category) => category.links) // Flattens all category link arrays into one list.
    .find((link) => link.id === linkId); // Returns the first link whose ID matches the requested ID.
}

function getTargetHref(target, role) { // Converts a route or link target into a browser-openable URL.
  if (!target) return null; // Rejects missing targets.

  if (typeof target === "string") { // Handles legacy string link IDs.
    return findRoleLinkById(role, target)?.href ?? null; // Looks up the ID in roleData and returns its href if found.
  }

  const href = target.path ?? target.href ?? null; // Uses routeRegistry path first, then legacy href if present.
  if (!href) return null; // Rejects targets with no URL.

  if (href.startsWith("/amids/routes/")) { // Detects local prototype registry routes.
    const routeBaseUrl = // Prepares the base URL used for local registry route pages.
      import.meta.env.VITE_AMIDS_ROUTE_BASE_URL ?? "http://127.0.0.1:3001"; // Uses an environment override or the local API server.
    return new URL(href, routeBaseUrl).href; // Converts the relative route path into a complete URL.
  }

  return href; // Returns real or already-complete URLs unchanged.
}

function detachWindowOpener(targetWindow) { // Removes opener access from a newly opened window.
  try { 
    targetWindow.opener = null; 
  } catch {}
}

function writeReservedRedirectPage(targetWindow, href, message, redirectDelayMs) { // Shows a loading page before redirecting a reserved tab.
  try { // Attempts to update the reserved tab while it is still same-origin.
    targetWindow.document.open(); // Clears the existing reserved-tab document.
    targetWindow.document.write(`
      <!doctype html>
      <title>Opening AMIDS route...</title>
      <body style="font-family: system-ui, sans-serif; padding: 24px;">
        <p id="opening-message"></p>
        <p><a id="manual-route-link">Open matched AMIDS route manually</a></p>
      </body>
    `); // Writes a small loading document without inserting route text as raw HTML.
    targetWindow.document.querySelector("#opening-message").textContent = message; // Inserts the route description safely as text.
    targetWindow.document.querySelector("#manual-route-link").href = href; // Provides a visible fallback link if redirect is slow.
    targetWindow.document.close(); // Finishes rendering the loading document.
    window.setTimeout(() => { // Gives the browser time to display the description before navigating.
      if (!targetWindow.closed) { // Avoids navigating a tab the user already closed.
        targetWindow.location.assign(href); // Redirects the reserved tab to the matched route.
      }
    }, redirectDelayMs); // Uses the requested short delay.
    return true; // Reports that the reserved tab was handled.
  } catch { // Handles browsers that block reserved-tab document access.
    return false; // Lets normal navigation handling try next.
  }
}

function navigateExistingWindow(targetWindow, href, options = {}) { // Navigates a pre-opened window to the final route URL.
  if (!targetWindow || targetWindow.closed) return false; // Fails if the reserved window is unavailable.

  try { // Attempts navigation while catching browser security errors.
    detachWindowOpener(targetWindow); // Removes opener access before navigating.
    if (options.loadingMessage) { // Checks whether the caller wants a visible loading message first.
      return writeReservedRedirectPage( // Shows the message and schedules the actual navigation.
        targetWindow,
        href,
        options.loadingMessage,
        options.redirectDelayMs ?? 1200
      );
    }
    targetWindow.location.assign(href); // Sends the reserved window to the selected route.
    return true; // Reports that navigation succeeded.
  } catch { // Handles navigation failures.
    return false; // Reports that navigation failed.
  }
}

function App() { // Defines the main React application component.
  const [activeRole, setActiveRole] = useState(null); // Stores the role currently being used.
  const [activeCategory, setActiveCategory] = useState(null); // Stores the selected information category.
  const [pendingRole, setPendingRole] = useState(null); // Stores the role being selected during transition.
  const [view, setView] = useState("roles"); // Tracks whether the UI shows role selection or the role interface.
  const [isTransitioning, setIsTransitioning] = useState(false); // Tracks whether a screen transition is in progress.
  const [assistantResult, setAssistantResult] = useState(null); // Stores the latest assistant recommendation.
  const [embeddedRoutes, setEmbeddedRoutes] = useState([]); // Stores routes displayed together in the iframe workspace.
  const [userProfile, setUserProfile] = useState(null); // Stores learned personalization data for the active role.
  const [expertRules, setExpertRules] = useState([]); // Stores only rules the user explicitly adds.

  const role = activeRole ? roleData[activeRole] : null; // Gets the full role object for the active role.
  const category = role && activeCategory ? role.categories[activeCategory] : null; // Gets the active category object if one is selected.
  const personalizedPreferences = useMemo( // Memoizes merged preferences so they update only when the profile changes.
    () => mergePreferencesWithProfile(defaultPreferences, userProfile), // Combines defaults with learned profile preferences.
    [userProfile] // Recomputes only when userProfile changes.
  );

  useEffect(() => { // Runs profile loading whenever the active role changes.
    if (!activeRole) { // Handles the state where no role is selected.
      setUserProfile(null); // Clears profile data when returning to the role picker.
      return undefined; // Ends the effect without cleanup work.
    }

    let ignore = false; // Tracks whether this async profile request has become outdated.

    loadUserProfile({ userId, roleKey: activeRole }).then((response) => { // Requests the saved profile for this user and role.
      if (!ignore && response.profile) { // Uses the response only if this effect is still current.
        setUserProfile(response.profile); // Saves the loaded personalization profile.
      }
    });

    return () => { // Defines cleanup that runs before the next effect.
      ignore = true; // Marks this request as stale if the role changes before it completes.
    };
  }, [activeRole]); // Runs the effect again whenever activeRole changes.

  function handleRoleChange(roleKey) { // Handles clicking Pilot, ATC, or Dispatcher.
    setPendingRole(roleKey); // Marks the clicked role as pending during animation.
    setIsTransitioning(true); // Starts the transition state.
    setActiveCategory(null); // Clears any previously selected category.
    setAssistantResult(null); // Clears the previous assistant recommendation.
    setEmbeddedRoutes([]); // Clears any displayed route iframe workspace.

    window.setTimeout(() => { // Waits briefly so the transition animation can play.
      setActiveRole(roleKey); // Makes the selected role active.
      setView("interface"); // Switches from role picker to role interface.
      setIsTransitioning(false); // Ends the transition state.
      setPendingRole(null); // Clears the temporary pending role.
    }, transitionDuration); // Uses the shared transition delay.
  }

  function resetView() { // Handles returning from a role interface to the three-role picker.
    setIsTransitioning(true); // Starts the exit transition.
    setActiveCategory(null); // Clears the active category.
    setAssistantResult(null); // Clears the assistant result.
    setEmbeddedRoutes([]); // Clears embedded route displays.

    window.setTimeout(() => { // Waits briefly before changing the screen.
      setActiveRole(null); // Clears the active role.
      setView("roles"); // Shows the role picker again.
      setIsTransitioning(false); // Ends the transition state.
    }, transitionDuration); // Uses the shared transition delay.
  }

  function handleOpenLink(target, options = {}) { // Opens a single route or link target.
    if (!role || !target) return false; // Fails when no role or target is available.

    const href = getTargetHref(target, role); // Converts the target into a URL.
    if (!href || href === "#") return false; // Rejects missing or placeholder links.

    if (navigateExistingWindow(options.targetWindow, href, options)) { // Uses a pre-opened tab when one exists.
      return true; // Reports success when the reserved tab was navigated.
    }

    const openedWindow = window.open(href, "_blank"); // Opens the route in a new browser tab.
    if (!openedWindow) return false; // Reports failure if the browser blocked the new tab.

    detachWindowOpener(openedWindow); // Removes opener access for security.
    return true; // Reports that the new tab opened successfully.
  }

  function toEmbeddedRoute(target) { // Converts a route into the format needed for iframe display.
    const href = getTargetHref(target, role); // Converts the route target into a URL.
    if (!href || href === "#") return null; // Skips invalid or placeholder URLs.

    return { // Returns a normalized embedded route object.
      id: target.id ?? href, // Uses the route ID, or the URL as a fallback ID.
      title: target.title ?? target.label ?? "AMIDS route", // Chooses the best available title.
      description: target.description ?? "", // Uses the route description or an empty string.
      href, // Stores the URL used by the iframe.
    };
  }

  function handleShowRoutes(targets) { // Displays multiple recommended routes on one screen.
    if (!role || !Array.isArray(targets)) return 0; // Rejects invalid input.

    const displayableRoutes = targets.map(toEmbeddedRoute).filter(Boolean).slice(0, 8); // Normalizes, filters, and limits routes to eight.
    setEmbeddedRoutes(displayableRoutes); // Saves the routes for the iframe workspace.
    return displayableRoutes.length; // Returns how many routes can be displayed.
  }

  return ( // Starts the JSX that describes the app UI.
    <main className="shell">
      {/* Renders the shared page header. */}
      <AppHeader />

      {/* Shows either the role picker or the active role interface. */}
      {view === "roles" ? (
        /* Renders the initial three large role buttons. */
        <RolePicker
          roles={roleData}
          selectedRole={pendingRole}
          isExiting={isTransitioning}
          onRoleChange={handleRoleChange}
        />
      ) : (
        /* Renders the selected role workspace. */
        <section
          className={`role-interface ${isTransitioning ? "is-exiting" : "is-entering"}`}
          aria-live="polite"
        >
          {/* Displays the selected role title and the back button. */}
          <div className="role-interface-topline">
            <div>
              <p className="eyebrow">Selected Role</p>
              <h2>{role.title}</h2>
            </div>
            <button className="clear-button" type="button" onClick={resetView}>
              Back to roles
            </button>
          </div>

          {/* Shows embedded route pages when the assistant recommends multiple routes. */}
          {embeddedRoutes.length > 0 ? (
            <section className="route-workspace" aria-label="Embedded AMIDS route workspace">
              {/* Displays the embedded workspace heading and return button. */}
              <div className="route-workspace-topline">
                <div>
                  <p className="eyebrow">Embedded Route Workspace</p>
                  <h3>{embeddedRoutes.length} recommended routes on one screen</h3>
                </div>
                <button className="clear-button" type="button" onClick={() => setEmbeddedRoutes([])}>
                  Back to assistant
                </button>
              </div>

              {/* Creates a responsive tile grid sized by the number of embedded routes. */}
              <div
                className={`route-tile-grid route-tile-grid-count-${embeddedRoutes.length}`}
              >
                {/* Creates one tile and iframe for each embedded route. */}
                {embeddedRoutes.map((route) => (
                  <article className="route-tile" key={`${route.id}-${route.href}`}>
                    <div className="route-tile-header">
                      <strong>{route.title}</strong>
                      <button type="button" onClick={() => handleOpenLink(route)}>
                        Open
                      </button>
                    </div>
                    <iframe src={route.href} title={route.title} />
                  </article>
                ))}
              </div>
            </section>
          ) : (
            /* Shows the normal assistant, rules, and manual control workspace. */
            <section className="workspace">
              {/* Lets users draft and save explicit assistant behavior rules. */}
              <RuleManagerPanel
                roleTitle={role.title}
                roleKey={activeRole}
                preferences={personalizedPreferences}
                expertRules={expertRules}
                onAddRule={(rule) => setExpertRules((currentRules) => [rule, ...currentRules])}
              />
              {/* Lets users ask for information and open routeRegistry results. */}
              <AssistantPanel
                userId={userId}
                roleKey={activeRole}
                preferences={personalizedPreferences}
                userProfile={userProfile}
                expertRules={expertRules}
                onOpenLink={handleOpenLink}
                onShowRoutes={handleShowRoutes}
                onProfileUpdated={setUserProfile}
                onAssistantResult={(result) => {
                  setAssistantResult(result);
                }}
              />
              {/* Shows manual category buttons and detailed roleData links. */}
              <ControlPanel
                role={role}
                category={category}
                categoryLabels={categoryLabels}
                activeCategory={activeCategory}
                recommendedLinkIds={assistantResult?.recommendedLinkIds ?? []}
                onCategoryChange={setActiveCategory}
              />
            </section>
          )}
        </section>
      )}
    </main>
  );
}

createRoot(document.querySelector("#root")).render( // Connects React to the root div in index.html and starts rendering.
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
