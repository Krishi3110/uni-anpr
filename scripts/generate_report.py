import json
import subprocess
import os

def generate_report():
    summary_path = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/outputs/reports/benchmark_summary.json"
    if not os.path.exists(summary_path):
        print("Benchmark summary not found. Run benchmark.py first.")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    prompt = f"""
    You are an AI assistant generating an engineering report for an ANPR system.
    The benchmark results are as follows:
    {json.dumps(summary, indent=2)}
    
    Please generate a professional engineering report summarizing these results, 
    evaluating if we met the targets (Precision > 90%, OCR accuracy > 85%, Latency < 1.0s),
    and providing recommendations for university parking integration.
    """
    
    print("Calling Ollama (qwen3:14b) to generate report...")
    try:
        res = subprocess.run(
            ["ollama", "run", "qwen3:14b"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding='utf-8'
        )
        if res.returncode == 0:
            report_text = res.stdout
            report_path = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/outputs/reports/final_report.md"
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {report_path}")
        else:
            print(f"Ollama failed: {res.stderr}")
    except Exception as e:
        print(f"Error calling Ollama: {e}")

if __name__ == "__main__":
    generate_report()
