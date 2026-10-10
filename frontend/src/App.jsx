import { Link, Route, Routes } from "react-router-dom";
import OverviewPage from "./pages/OverviewPage.jsx";
import SegmentsPage from "./pages/SegmentsPage.jsx";
import { apiBaseUrl } from "./api/client.js";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="brand">Customer Segmentation</p>
          <p className="tagline">Banking Behavioral Intelligence</p>
        </div>
        <nav>
          <Link to="/">Overview</Link>
          <Link to="/segments">Segments</Link>
        </nav>
      </header>
      <main>
        <p className="api-hint">
          API base: <code>{apiBaseUrl}</code>
        </p>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/segments" element={<SegmentsPage />} />
        </Routes>
      </main>
    </div>
  );
}
