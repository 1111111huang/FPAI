import { MatchAnalysisPage } from "@/components/MatchUI";

export default function Page({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: { home?: string; away?: string; date?: string; league?: string };
}) {
  return (
    <MatchAnalysisPage
      id={params.id}
      home={searchParams.home ?? ""}
      away={searchParams.away ?? ""}
      date={searchParams.date ?? ""}
      league={searchParams.league ?? "E0"}
    />
  );
}
