import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import ProblemSection from "@/components/ProblemSection";
import PipelineSection from "@/components/PipelineSection";
import ResultsSection from "@/components/ResultsSection";
import MemoryExplorer from "@/components/MemoryExplorer";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Nav />
      <Hero />
      <ProblemSection />
      <PipelineSection />
      <ResultsSection />
      <MemoryExplorer />
      <Footer />
    </main>
  );
}
