function QuickMetric({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function RoleSummary({ role }) {
  return (
    <aside className="role-summary">
      <p className="eyebrow">Current Role</p>
      <h2>{role.title}</h2>
      <p>{role.description}</p>
      <div className="quick-metrics" aria-label="Current example weather metrics">
        {role.metrics.map(([label, value]) => (
          <QuickMetric key={label} label={label} value={value} />
        ))}
      </div>
    </aside>
  );
}
