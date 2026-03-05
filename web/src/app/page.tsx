import GlobeExperience from "./GlobeExperience";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#050712] text-slate-100">
      <div className="flex flex-col md:flex-row h-screen">
        <section className="flex-1 relative">
          <GlobeExperience />
        </section>

        <aside className="w-full md:w-[360px] border-t md:border-t-0 md:border-l border-white/5 bg-white/5/10 backdrop-blur-xl">
          <div className="p-4 md:p-6 h-full flex flex-col">
            <h1 className="text-xl font-semibold tracking-tight mb-2">
              Sky Sentry
            </h1>
            <p className="text-sm text-slate-400 mb-6">
              Live 3D flight tracking on a holographic Earth powered by
              OpenSky.
            </p>
            <div className="flex-1 rounded-2xl bg-white/5 border border-white/10 p-4 overflow-auto">
              <p className="text-xs text-slate-400">
                Hover or click a jet to see details here.
              </p>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}
