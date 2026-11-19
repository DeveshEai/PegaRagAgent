"""
Pega Academy - Complete 40+ Hands-on Challenges Downloader
==========================================================
Downloads all official challenge walk-through guides, hands-on exercises,
and solutions across the Lead System Architect v7 course as clean PDFs!
"""

import asyncio, os, re, sys, io, time
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv(Path(__file__).parent / ".env")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent / "Lead System Architect"
COOKIE_NAME = os.environ.get("PEGA_SESSION_COOKIE_NAME", "SSESS7320634c8bd6f7dc1a41435835961d4c").strip()
COOKIE_VAL = os.environ.get("PEGA_SESSION_COOKIE", "").strip()
TA_VAL = os.environ.get("PEGA_TA_SESSION_ID", "").strip()
HELLO_DATA = os.environ.get("PEGA_HELLO_DATA", "").strip()

ALL_CHALLENGES = [
    # 02_Planning_and_Design
    ("02_Planning_and_Design", "01_designing-case-hierarchy", "https://academy.pega.com/challenge/designing-case-hierarchy/v6"),
    ("02_Planning_and_Design", "02_designing-subcase-hierarchy", "https://academy.pega.com/challenge/designing-subcase-hierarchy/v6"),
    ("02_Planning_and_Design", "03_creating-delivery-service-and-mdc-applications", "https://academy.pega.com/challenge/creating-delivery-service-and-mdc-applications/v3"),
    ("02_Planning_and_Design", "04_designing-case-specialization-and-extensibility", "https://academy.pega.com/challenge/designing-case-specialization-and-extensibility/v6"),
    ("02_Planning_and_Design", "05_creating-rest-integration-interface", "https://academy.pega.com/challenge/creating-rest-integration-interface/v7"),
    ("02_Planning_and_Design", "06_designing-options-invoice-case-handling", "https://academy.pega.com/challenge/designing-options-invoice-case-handling/v3"),
    ("02_Planning_and_Design", "07_implementing-effective-data-model", "https://academy.pega.com/challenge/implementing-effective-data-model/v7"),
    ("02_Planning_and_Design", "08_authoring-data-model-and-associated-views-using-app-studio", "https://academy.pega.com/challenge/authoring-data-model-and-associated-views-using-app-studio/v2"),
    ("02_Planning_and_Design", "09_implementing-data-instance-first-design-pattern", "https://academy.pega.com/challenge/implementing-data-instance-first-design-pattern/v2"),
    ("02_Planning_and_Design", "10_defining-reporting-strategy", "https://academy.pega.com/challenge/defining-reporting-strategy/v6"),
    ("02_Planning_and_Design", "11_extracting-case-data-using-bix", "https://academy.pega.com/challenge/extracting-case-data-using-bix/v2"),
    ("02_Planning_and_Design", "12_creating-insight-executive-manager", "https://academy.pega.com/challenge/creating-insight-executive-manager/v2"),
    ("02_Planning_and_Design", "13_configuring-openid-connect-authentication-using-app-studio", "https://academy.pega.com/challenge/configuring-openid-connect-authentication-using-app-studio/v2"),
    ("02_Planning_and_Design", "14_configuring-saml-authentication-using-app-studio", "https://academy.pega.com/challenge/configuring-saml-authentication-using-app-studio/v2"),
    ("02_Planning_and_Design", "15_defining-authorization-scheme", "https://academy.pega.com/challenge/defining-authorization-scheme/v7"),

    # 03_Constellation_Adoption
    ("03_Constellation_Adoption", "01_designing-views-constellation", "https://academy.pega.com/challenge/designing-views-constellation/v3"),
    ("03_Constellation_Adoption", "02_creating-field-groups-constellation-views", "https://academy.pega.com/challenge/creating-field-groups-constellation-views/v3"),
    ("03_Constellation_Adoption", "03_configuring-conditional-visibility-constellation-views", "https://academy.pega.com/challenge/configuring-conditional-visibility-constellation-views/v3"),
    ("03_Constellation_Adoption", "04_creating-partial-views-constellation", "https://academy.pega.com/challenge/creating-partial-views-constellation/v3"),
    ("03_Constellation_Adoption", "05_designing-complex-views-constellation", "https://academy.pega.com/challenge/designing-complex-views-constellation/v3"),
    ("03_Constellation_Adoption", "06_linking-cases-and-case-types-constellation-applications", "https://academy.pega.com/challenge/linking-cases-and-case-types-constellation-applications/v3"),
    ("03_Constellation_Adoption", "07_assigning-primary-fields-case-tabs-and-more-constellation", "https://academy.pega.com/challenge/assigning-primary-fields-case-tabs-and-more-constellation/v3"),
    ("03_Constellation_Adoption", "08_adding-portals-and-landing-pages-constellation-applications", "https://academy.pega.com/challenge/adding-portals-and-landing-pages-constellation-applications/v3"),
    ("03_Constellation_Adoption", "09_blending-traditional-ui-and-constellation", "https://academy.pega.com/challenge/blending-traditional-ui-and-constellation/v3"),
    ("03_Constellation_Adoption", "10_exploring-dx-api", "https://academy.pega.com/challenge/exploring-dx-api/v3"),
    ("03_Constellation_Adoption", "11_debugging-constellation-applications", "https://academy.pega.com/challenge/debugging-constellation-applications/v3"),

    # 04_Infrastructure_and_Architecture
    ("04_Infrastructure_and_Architecture", "01_designing-pega-enterprise", "https://academy.pega.com/challenge/designing-pega-enterprise/v6"),
    ("04_Infrastructure_and_Architecture", "02_creating-case-kafka-data", "https://academy.pega.com/challenge/creating-case-kafka-data/v2"),
    ("04_Infrastructure_and_Architecture", "03_creating-case-file-data-set", "https://academy.pega.com/challenge/creating-case-file-data-set/v2"),
    ("04_Infrastructure_and_Architecture", "04_using-database-index-report-performance", "https://academy.pega.com/challenge/using-database-index-report-performance/v6"),

    # 05_Implementation_and_Deployment
    ("05_Implementation_and_Deployment", "01_designing-background-processing", "https://academy.pega.com/challenge/designing-background-processing/v7"),
    ("05_Implementation_and_Deployment", "02_reusing-standard-queue-processor", "https://academy.pega.com/challenge/reusing-standard-queue-processor/v6"),
    ("05_Implementation_and_Deployment", "03_implementing-web-embed-interfaces", "https://academy.pega.com/challenge/implementing-web-embed-interfaces/v3"),
    ("05_Implementation_and_Deployment", "04_configuring-work-landing-page-show-worklists-and-work-queues", "https://academy.pega.com/challenge/configuring-work-landing-page-show-worklists-and-work-queues/v2"),
    ("05_Implementation_and_Deployment", "05_creating-landing-pages", "https://academy.pega.com/challenge/creating-landing-pages/v2"),
    ("05_Implementation_and_Deployment", "06_enhancing-pricing-data-model", "https://academy.pega.com/challenge/enhancing-pricing-data-model/v7"),
    ("05_Implementation_and_Deployment", "07_simulating-truck-rating-response", "https://academy.pega.com/challenge/simulating-truck-rating-response/v3"),
    ("05_Implementation_and_Deployment", "08_identifying-and-mitigating-security-risks", "https://academy.pega.com/challenge/identifying-and-mitigating-security-risks/v7"),
    ("05_Implementation_and_Deployment", "09_analyzing-and-extracting-data-documents-using-pega-genai", "https://academy.pega.com/challenge/analyzing-and-extracting-data-documents-using-pega-genai/v1"),
    ("05_Implementation_and_Deployment", "10_optimizing-case-processing-pega-genai-agents", "https://academy.pega.com/challenge/optimizing-case-processing-pega-genai-agents/v1")
]

async def main():
    print("=" * 65)
    print("  PEGA ACADEMY - ALL 40 HANDS-ON CHALLENGES PDF DOWNLOADER")
    print("=" * 65)

    if not COOKIE_VAL:
        print("Error: PEGA_SESSION_COOKIE is not set. Please configure it in .env before running the downloader.")
        return

    total = len(ALL_CHALLENGES)
    print(f"\nTotal challenges to download: {total}\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 1000})
        cookies = [
            {"name": COOKIE_NAME, "value": COOKIE_VAL, "domain": ".academy.pega.com", "path": "/"},
            {"name": "PERSISTENT_LOGIN_PEGASSO_PROD", "value": "1", "domain": ".pega.com", "path": "/"}
        ]
        if TA_VAL:
            cookies.append({"name": "TAsessionID", "value": TA_VAL, "domain": ".academy.pega.com", "path": "/"})
        if HELLO_DATA:
            cookies.append({"name": "hellodata", "value": HELLO_DATA, "domain": ".pega.com", "path": "/"})
        await context.add_cookies(cookies)
        page = await context.new_page()

        success = 0
        for idx, (section, slug, url) in enumerate(ALL_CHALLENGES, 1):
            folder = BASE_DIR / section / "Challenges"
            folder.mkdir(parents=True, exist_ok=True)
            pdf_path = folder / f"{slug}.pdf"

            if pdf_path.exists() and pdf_path.stat().st_size > 60000:
                print(f"[{idx}/{total}] ⏭ Already exists: {pdf_path.name}")
                success += 1
                continue

            print_url = url.rstrip("/") + "/print"
            print(f"[{idx}/{total}] 📥 Downloading Challenge: {pdf_path.name} ...")

            try:
                await page.goto(print_url, wait_until="domcontentloaded", timeout=40000)
                await page.wait_for_timeout(2000)
                await page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "20px", "bottom": "20px", "left": "20px", "right": "20px"}
                )
                kb = pdf_path.stat().st_size / 1024
                print(f"      ✅ Saved! ({kb:.1f} KB)")
                success += 1
            except Exception as e:
                print(f"      ❌ Failed: {e}")

            # Polite delay to avoid Cloudflare rate limiting
            await asyncio.sleep(1.5)

        await browser.close()

    print("\n" + "=" * 65)
    print(f"🎉 Successfully downloaded {success}/{total} Hands-on Challenges!")
    print(f"📁 Location: {BASE_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(main())
