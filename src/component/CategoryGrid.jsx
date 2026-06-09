function CategoryButton({ label, summary, active, onClick }) {
  return (
    <button
      className={`category-button ${active ? "is-active" : ""}`}
      type="button"
      onClick={onClick}
    >
      <strong>{label}</strong>
      <span>{summary}</span>
    </button>
  );
}

export default function CategoryGrid({
  categories,
  categoryLabels,
  activeCategory,
  onCategoryChange,
}) {
  return (
    <div className="category-grid">
      {Object.entries(categories).map(([categoryKey, category]) => (
        <CategoryButton
          active={categoryKey === activeCategory}
          key={categoryKey}
          label={categoryLabels[categoryKey]}
          summary={category.summary}
          onClick={() => onCategoryChange(categoryKey)}
        />
      ))}
    </div>
  );
}
