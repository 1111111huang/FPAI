import { MatchAnalysisDraft } from "@/components/sandbox/DraftUI";

export default function Page({ params }: { params: { id: string } }) {
  return <MatchAnalysisDraft id={params.id} />;
}
