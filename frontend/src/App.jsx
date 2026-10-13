import { NavLink, Route, Routes } from "react-router-dom";
import { getHealth } from "./api/client.js";
import StatusPill from "./components/StatusPill.jsx";
import { useAsync } from "./hooks/useAsync.js";
import ChampionPage from "./pages/ChampionPage.jsx";
import ComparePage from "./pages/ComparePage.jsx";
import DecisionsPage from "./pages/DecisionsPage.jsx";
import EdaPage from "./pages/EdaPage.jsx";
import HomePage from "./pages/HomePage.jsx";
import ScorePage from "./pages/ScorePage.jsx";
import SegmentDetailPage from "./pages/SegmentDetailPage.jsx";
import SegmentsPage from "./pages/SegmentsPage.jsx";

const NAV = [
  ["/", "Home", true],
  ["/eda", "Data"],
  ["/champion", "Why"],
  ["/compare", "Compare"],
  ["/segments", "Segments"],
  ["/decisions", "Applications"],
  ["/score", "Demo"],
];

export default function App() {
  const { data: health, error, loading } = useAsync(() => getHealth(), []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="brand">Behavioral Banking Intelligence</p>
          <p className="brand-sub">Berka champion · provisional</p>
        </div>
        <nav aria-label="Primary">
          {NAV.map(([to, label, end]) => (
            <NavLink
              key={to}
              to={to}
              end={Boolean(end)}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <StatusPill health={health} loading={loading} error={error} />
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/champion" element={<ChampionPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/segments" element={<SegmentsPage />} />
          <Route path="/segments/:id" element={<SegmentDetailPage />} />
          <Route path="/decisions" element={<DecisionsPage />} />
          <Route path="/eda" element={<EdaPage />} />
          <Route path="/score" element={<ScorePage />} />
        </Routes>
      </main>
    </div>
  );
}
