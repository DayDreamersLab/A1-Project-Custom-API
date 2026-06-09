import { useState } from "react";

const exampleExpertRules = [
  "When I ask for runway wind data, prioritize wind shear information related to runway operations.",
  "When I ask about LVP, show RVR and low visibility procedure first.",
];

function createRuleFromInput(ruleText, roleKey) {
  const normalizedRule = ruleText.trim();
  const triggerMatch = normalizedRule.match(/(?:if|when)\s+i\s+(?:say|ask|type|request)\s+["']?([^"',.]+)["']?/i);

  return {
    id: crypto.randomUUID(),
    name: triggerMatch ? `User rule for "${triggerMatch[1].trim()}"` : "User-defined expert rule",
    trigger: triggerMatch?.[1]?.trim() ?? normalizedRule.slice(0, 48),
    instruction: normalizedRule,
    linkIds: [],
    scope: roleKey,
    status: "draft",
  };
}

export default function RuleManagerPanel({ roleTitle, roleKey, preferences, expertRules, onAddRule }) {
  const [ruleText, setRuleText] = useState("");
  const [draftRule, setDraftRule] = useState(null);

  function handleDraftRule(event) {
    event.preventDefault();
    if (!ruleText.trim()) return;

    // FUTURE AI RULE PARSER:
    // Replace createRuleFromInput() with a secure backend call that asks an AI model
    // to convert the user's natural-language rule into structured fields:
    // trigger, instruction, role scope, category scope, affected link IDs, and conflicts.
    setDraftRule(createRuleFromInput(ruleText, roleKey));
  }

  function handleSaveRule() {
    if (!draftRule) return;

    onAddRule({
      ...draftRule,
      status: "active",
    });
    setDraftRule(null);
    setRuleText("");
  }

  return (
    <section className="rule-manager-panel" aria-labelledby="rule-manager-title">
      <div className="assistant-panel-header">
        <div>
          <p className="eyebrow">Rules & Preferences</p>
          <h3 id="rule-manager-title">{roleTitle} Control Layer</h3>
        </div>
        <span className="assistant-mode">explicit-rules</span>
      </div>

      <form className="assistant-query rule-query" onSubmit={handleDraftRule}>
        <label htmlFor="rule-input">Tell the assistant how it should behave next time</label>
        <textarea
          id="rule-input"
          value={ruleText}
          onChange={(event) => setRuleText(event.target.value)}
          placeholder="Example: When I ask about LVP, show RVR and low visibility procedure first."
          rows="5"
        />
        <button type="submit">Draft rule</button>
      </form>

      {draftRule && (
        <article className="assistant-result">
          <h4>Draft rule awaiting approval</h4>
          <p>Trigger: {draftRule.trigger}</p>
          <p>Instruction: {draftRule.instruction}</p>
          <div className="assistant-feedback">
            <button type="button" onClick={handleSaveRule}>
              Save rule
            </button>
            <button type="button" onClick={() => setDraftRule(null)}>
              Cancel
            </button>
          </div>
        </article>
      )}

      <div className="assistant-rules-grid">
        <section>
          <h4>Personalization</h4>
          <p>Usual airports: {preferences.usualAirports}</p>
          <p>Preferred categories: {preferences.preferredCategories.join(", ")}</p>
        </section>
        <section>
          <h4>Example Rules</h4>
          <p>These examples are not active until typed, drafted, and saved.</p>
          {exampleExpertRules.map((exampleRule) => (
            <p className="example-rule" key={exampleRule}>
              {exampleRule}
            </p>
          ))}
        </section>
        <section>
          <h4>Active Expert Rules</h4>
          {expertRules.length > 0 ? (
            expertRules.map((rule) => (
              <p key={rule.id}>{rule.name}</p>
            ))
          ) : (
            <p>No expert rules added yet.</p>
          )}
        </section>
      </div>
    </section>
  );
}
