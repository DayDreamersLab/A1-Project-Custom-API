import CategoryGrid from "./CategoryGrid";
import DetailPanel from "./DetailPanel";

export default function ControlPanel({
  role,
  category,
  categoryLabels,
  activeCategory,
  recommendedLinkIds,
  onCategoryChange,
}) {
  return (
    <section className="control-panel" aria-labelledby="panel-title">
      <div className="panel-topline">
        <div>
          <p className="eyebrow">Information Categories</p>
          <h2 id="panel-title">{role.panelTitle}</h2>
        </div>
      </div>

      <CategoryGrid
        categories={role.categories}
        categoryLabels={categoryLabels}
        activeCategory={activeCategory}
        onCategoryChange={onCategoryChange}
      />

      <DetailPanel
        role={role}
        category={category}
        categoryLabel={activeCategory ? categoryLabels[activeCategory] : null}
        recommendedLinkIds={recommendedLinkIds}
      />
    </section>
  );
}
