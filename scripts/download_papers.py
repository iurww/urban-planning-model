"""Download all papers in paperlist to papers/downloads/."""

import re
import time
import urllib.request
from pathlib import Path

PAPERS = [
    # ── 土地规划 / 用地配置生成 ─────────────────────────────────────────────────
    ("Spatial Planning of Urban Communities via Deep Reinforcement Learning",
     "https://fi.ee.tsinghua.edu.cn/public/publications/77681512-6436-11ee-84fe-0242ac120002.pdf"),
    ("Human-Instructed Deep Hierarchical Generative Learning for Automated Urban Planning",
     "https://arxiv.org/pdf/2212.00904"),
    ("Automated Urban Planning for Reimagining City Configuration via Adversarial Learning Quantification Generation and Evaluation",
     "https://arxiv.org/pdf/2112.14699"),
    ("Reimagining City Configuration Automated Urban Planning via Adversarial Learning",
     "https://arxiv.org/pdf/2008.09912"),
    ("AI Agent as Urban Planner Consensus-based Multi-Agent Reinforcement Learning",
     "https://arxiv.org/pdf/2310.16772"),
    ("Evaluation and Development of Sustainable Urban Land Use Plans through Spatial Optimization",
     "http://eprints.gla.ac.uk/185457/1/185457.pdf"),
    # ── 土地规划 / 城市形态与街区生成 ──────────────────────────────────────────
    ("MetroGAN Simulating Urban Morphology with Generative Adversarial Network",
     "https://arxiv.org/pdf/2207.02590"),
    # ── 土地规划 / LLM ───────────────────────────────────────────────────────
    ("Large Language Model for Participatory Urban Planning",
     "https://arxiv.org/pdf/2402.17161"),
    ("Planning Living and Judging A Multi-agent LLM-based Framework for Cyclical Urban Planning",
     "https://arxiv.org/pdf/2412.20505"),
    ("UrbanPlanBench A Comprehensive Urban Planning Benchmark for Evaluating Large Language Models",
     "https://arxiv.org/pdf/2504.21027"),
    # ── 道路规划 / 规划与生成 ───────────────────────────────────────────────────
    ("Road Planning for Slums via Deep Reinforcement Learning",
     "https://arxiv.org/pdf/2305.13060"),
    ("MetroGNN Metro Network Expansion with Reinforcement Learning",
     "https://arxiv.org/pdf/2403.09197"),
    ("Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning",
     "https://arxiv.org/pdf/2404.05894"),
    ("Applications of Deep Reinforcement Learning to Urban Transit Network Design",
     "https://arxiv.org/pdf/2502.17758"),
    ("Incorporating Planning Intelligence into Deep Learning A Planning Support Tool for Street Network Design",
     "https://arxiv.org/pdf/2010.04536"),
    # ── 道路规划 / 路网分析 ────────────────────────────────────────────────────
    ("Quantifying the Spatial Homogeneity of Urban Road Networks via Graph Neural Networks",
     "https://arxiv.org/pdf/2101.00307"),
    # ── 道路规划 / LLM ───────────────────────────────────────────────────────
    ("USTBench Benchmarking and Dissecting Spatiotemporal Reasoning of LLMs as Urban Agents",
     "https://arxiv.org/pdf/2505.17572"),
    # ── POI选址 / 小模型 ───────────────────────────────────────────────────────
    ("Deconstructing Laws of Accessibility and Facility Distribution in Cities",
     "https://arxiv.org/pdf/2007.08832"),
    ("Reinforcement Learning-based Placement of Charging Stations in Urban Road Networks",
     "https://arxiv.org/pdf/2206.06011"),
    ("Swap-based Deep Reinforcement Learning for Facility Location Problems in Networks",
     "https://arxiv.org/pdf/2312.15658"),
    ("Large-scale Urban Facility Location Selection with Knowledge-informed Reinforcement Learning",
     "https://arxiv.org/pdf/2409.01588"),
    # ── POI选址 / LLM ────────────────────────────────────────────────────────
    ("AgentMove A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction",
     "https://arxiv.org/pdf/2408.13986"),
    # ── 红绿灯控制 / 小模型 ────────────────────────────────────────────────────
    ("CoLight Learning Network-level Cooperation for Traffic Signal Control",
     "https://arxiv.org/pdf/1905.05717"),
    ("MTLight Efficient Multi-Task Reinforcement Learning for Traffic Signal Control",
     "https://arxiv.org/pdf/2404.00886"),
    ("DiffLight A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data",
     "https://arxiv.org/pdf/2410.22938"),
    ("CityLight A Neighborhood-inclusive Universal Model for Coordinated City-scale Traffic Signal Control",
     "https://arxiv.org/pdf/2406.02126"),
    # ── 红绿灯控制 / LLM ─────────────────────────────────────────────────────
    ("LLMLight Large Language Models as Traffic Signal Control Agents",
     "https://arxiv.org/pdf/2312.16044"),
    ("LLM-Assisted Light Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control",
     "https://arxiv.org/pdf/2403.08337"),
    ("CollmLight Cooperative Large Language Model Agents for Network-wide Traffic Signal Control",
     "https://arxiv.org/pdf/2503.11739"),
    ("VLMLight Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning",
     "https://arxiv.org/pdf/2505.19486"),
    # ── 路径规划 / 小模型 ──────────────────────────────────────────────────────
    ("Towards Alleviating Traffic Congestion Optimal Route Planning for Massive-Scale Trips",
     "https://www.ijcai.org/proceedings/2020/0470.pdf"),
    ("Attention Learn to Solve Routing Problems",
     "https://arxiv.org/pdf/1803.08475"),
    # ── 路径规划 / LLM ───────────────────────────────────────────────────────
    ("LLM-A* Large Language Model Enhanced Incremental Heuristic Search on Path Planning",
     "https://arxiv.org/pdf/2407.02511"),
    ("Perceive Reflect and Plan Designing LLM Agent for Goal-Directed City Navigation without Instructions",
     "https://arxiv.org/pdf/2408.04168"),
    ("LLMAP LLM-Assisted Multi-Objective Route Planning with User Preferences",
     "https://arxiv.org/pdf/2509.12273"),
    # ── 背景：15分钟城市 ───────────────────────────────────────────────────────
    ("Introducing the 15-Minute City Sustainability Resilience and Place Identity in Future Post-Pandemic Cities",
     "https://www.mdpi.com/2624-6511/4/1/6/pdf"),
    ("Barcelona under the 15-Minute City Lens Mapping the Accessibility and Proximity Potential",
     "https://www.mdpi.com/2624-6511/5/1/10/pdf"),
    ("The 15-Minute City Concept Can Shape a Net-Zero Urban Future",
     "https://www.nature.com/articles/s41599-022-01145-0.pdf"),
    # ── 综述 ─────────────────────────────────────────────────────────────────
    ("Generative Adversarial Networks in the Built Environment A Comprehensive Review",
     "https://filipbiljecki.com/publications/2022_bae_gan.pdf"),
]

OUT_DIR = Path(__file__).parent.parent / "papers" / "downloads"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def sanitize(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "", title)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] + ".pdf"


def download(title: str, url: str) -> None:
    path = OUT_DIR / sanitize(title)
    if path.exists():
        print(f"[skip] {title}")
        return
    print(f"[dl  ] {title}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 1000 or not data.startswith(b"%PDF"):
            print(f"  [warn] not a valid PDF ({len(data)} bytes)")
            return
        path.write_bytes(data)
        print(f"  [ok  ] saved {len(data)//1024} KB → {path.name}")
    except Exception as exc:
        print(f"  [err ] {exc}")


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    for title, url in PAPERS:
        download(title, url)
        time.sleep(1)
    print("\nDone.")
