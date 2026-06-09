export default function RolePicker({ roles, selectedRole, isExiting, onRoleChange }) {
  return (
    <section
      className={`role-picker no-role-selected ${isExiting ? "is-exiting" : "is-entering"}`}
      aria-label="Select aviation role"
    >
      {Object.entries(roles).map(([roleKey, role]) => (
        <article
          className={`role-card ${selectedRole === roleKey ? "is-selected" : ""}`}
          key={roleKey}
        >
          <button
            className="role-button"
            type="button"
            disabled={isExiting}
            onClick={() => onRoleChange(roleKey)}
          >
            <span>{role.title}</span>
            <small>{role.shortLabel}</small>
          </button>
        </article>
      ))}
    </section>
  );
}
