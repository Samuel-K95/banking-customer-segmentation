export default function PageShell({ title, lead, children, actions }) {
  return (
    <div className="page">
      <header className="section-head">
        <div className="section-head-row">
          <div>
            <h2>{title}</h2>
            {lead ? <p>{lead}</p> : null}
          </div>
          {actions ? <div className="section-actions">{actions}</div> : null}
        </div>
      </header>
      {children}
    </div>
  );
}
