"""
Pega Academy - Lead System Architect v7 Complete PDF Downloader
===============================================================
Downloads all 27 modules of Lead System Architect v7 as official clean PDFs.
"""

import asyncio, os, re, sys, io, time
from pathlib import Path
from playwright.async_api import async_playwright

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent / "Lead System Architect"
BASE_DIR.mkdir(parents=True, exist_ok=True)

COOKIE_VAL = "smJ0VPW0ig9UFhgDt1DijSyritSR0w9eKL0zU47zCxul231J"

# COMPLETE CURRICULUM OF LEAD SYSTEM ARCHITECT (v7)
CURRICULUM = {
    "01_LSA_Mindset": [
        "https://academy.pega.com/module/lsa-mindset/v6"
    ],
    "02_Planning_and_Design": [
        "https://academy.pega.com/module/application-structure-design/v3",
        "https://academy.pega.com/module/designing-specialization/v6",
        "https://academy.pega.com/module/designing-data-models-and-reporting-strategies/v1",
        "https://academy.pega.com/module/api-design/v2",
        "https://academy.pega.com/module/authentication-design/v1",
        "https://academy.pega.com/module/authorization-design/v1",
        "https://academy.pega.com/module/application-monitoring-pdc/v1",
        "https://academy.pega.com/module/design-patterns-and-strategies-scalable-deployment/v1"
    ],
    "03_Constellation_Adoption": [
        "https://academy.pega.com/module/fundamental-concepts-constellation/v2",
        "https://academy.pega.com/module/modeling-data-constellation-applications/v2",
        "https://academy.pega.com/module/ui-authoring-constellation/v3",
        "https://academy.pega.com/module/extending-ui-options/v3",
        "https://academy.pega.com/module/tools-debugging-constellation-applications/v3"
    ],
    "04_Infrastructure_and_Architecture": [
        "https://academy.pega.com/module/enterprise-architecture/v3",
        "https://academy.pega.com/module/deployment-architecture/v1",
        "https://academy.pega.com/module/pega-genai-architecture/v1",
        "https://academy.pega.com/module/externalization-services/v1",
        "https://academy.pega.com/module/enterprise-security/v1",
        "https://academy.pega.com/module/system-administration-basics/v1",
        "https://academy.pega.com/module/reporting-data-architecture-and-performance-optimization/v1"
    ],
    "05_Implementation_and_Deployment": [
        "https://academy.pega.com/module/advanced-background-processing/v7",
        "https://academy.pega.com/module/work-assignment/v6",
        "https://academy.pega.com/module/handling-flow-changes-cases-flight/v6",
        "https://academy.pega.com/module/secure-application-design-and-governance/v1",
        "https://academy.pega.com/module/testing-design/v2",
        "https://academy.pega.com/module/deployment-design/v2",
        "https://academy.pega.com/module/pega-genai-implementation/v1"
    ]
}

def sanitize(name):
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip(". ")
    return name[:90] or "module"

async def main():
    print("=" * 65)
    print("  PEGA ACADEMY - LEAD SYSTEM ARCHITECT COMPLETE PDF EXTRACTOR")
    print("=" * 65)

    total_modules = sum(len(mods) for mods in CURRICULUM.values())
    print(f"\nTotal modules to download across all 5 sections: {total_modules}\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1400, "height": 1000}
        )
        await context.add_cookies([
            {"name": "SSESS7320634c8bd6f7dc1a41435835961d4c", "value": COOKIE_VAL, "domain": ".academy.pega.com", "path": "/"}
        ])
        page = await context.new_page()

        overall_idx = 0
        success_count = 0

        for section_name, modules in CURRICULUM.items():
            folder = BASE_DIR / section_name
            folder.mkdir(parents=True, exist_ok=True)
            print(f"\n📂 Section: {section_name} ({len(modules)} modules)")

            for idx, mod_url in enumerate(modules, 1):
                overall_idx += 1
                mod_slug = mod_url.split("/")[4]
                pdf_filename = f"{idx:02d}_{sanitize(mod_slug)}.pdf"
                dest_path = folder / pdf_filename

                if dest_path.exists() and dest_path.stat().st_size > 50000:
                    print(f"   [{overall_idx}/{total_modules}] ⏭ Already downloaded: {pdf_filename}")
                    success_count += 1
                    continue

                print_url = mod_url.rstrip("/") + "/print"
                print(f"   [{overall_idx}/{total_modules}] 📥 Exporting PDF: {pdf_filename} ...")

                try:
                    await page.goto(print_url, wait_until="domcontentloaded", timeout=45000)
                    await page.wait_for_timeout(2500)

                    await page.pdf(
                        path=str(dest_path),
                        format="A4",
                        print_background=True,
                        margin={"top": "25px", "bottom": "25px", "left": "25px", "right": "25px"}
                    )
                    size_kb = dest_path.stat().st_size / 1024
                    print(f"      ✅ Saved! ({size_kb:.1f} KB)")
                    success_count += 1
                except Exception as e:
                    print(f"      ❌ Error on {print_url}: {e}")

                await asyncio.sleep(1)

        await browser.close()

        print("\n" + "=" * 65)
        print(f"🎉 COMPLETED! Successfully downloaded {success_count}/{total_modules} module PDFs!")
        print(f"📁 Folder: {BASE_DIR}")
        print("=" * 65)

    input("\nPress ENTER to close...")

if __name__ == "__main__":
    asyncio.run(main())
