export default function OfflinePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 bg-page px-6 text-center">
      <h1 className="text-xl font-semibold text-ink">You&apos;re offline</h1>
      <p className="max-w-sm text-sm text-ink-secondary">
        Oddsey needs a connection to load fixtures and recommendations. Reconnect and try again.
      </p>
    </div>
  );
}
