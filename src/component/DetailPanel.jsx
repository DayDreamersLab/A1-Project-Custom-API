function NavigationLink({ link, roleTitle, isRecommended }) {
  function handleClick(event) {
    if (link.href === "#") {
      event.preventDefault();
    }
  }

  return (
    <a
      className={`briefing-link ${link.priority} ${isRecommended ? "is-recommended" : ""}`}
      href={link.href}
      target={link.href === "#" ? undefined : "_blank"}
      rel={link.href === "#" ? undefined : "noreferrer"}
      onClick={handleClick}
      aria-label={`${link.label} for ${roleTitle}`}
    >
      <strong>{link.label}</strong>
      <span>{link.description}</span>
    </a>
  );
}

export default function DetailPanel({ role, category, categoryLabel, recommendedLinkIds = [] }) {
  if (!category) {
    return (
      <article className="detail-panel detail-panel-empty">
        <p className="eyebrow">{role.title}</p>
        <h3>Choose a weather category</h3>
        <p>
          Select Wind, Precipitation, Visibility, Temperature, Altimeter / QNH,
          or Volcanic Ash to open the detailed briefing for this role.
        </p>
      </article>
    );
  }

  return (
    <article className="detail-panel">
      <p className="eyebrow">{role.title}</p>
      <h3>{categoryLabel}</h3>
      <p>{category.detail}</p>
      <div className="link-grid">
        {category.links.map((link) => (
          <NavigationLink
            isRecommended={recommendedLinkIds.includes(link.id)}
            link={link}
            key={link.id}
            roleTitle={role.title}
          />
        ))}
      </div>
    </article>
  );
}
