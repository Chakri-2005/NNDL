import os
import asyncio
from playwright.async_api import async_playwright

async def take_screenshots():
    mockups_dir = os.path.abspath("slides/ui_mockups")
    screenshots_dir = os.path.join(mockups_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    html_files = [
        "capture_screen.html",
        "result_screen.html",
        "abstain_screen.html",
        "tracker_screen.html"
    ]
    
    async with async_playwright() as p:
        # 375x812 viewport (iPhone X/11/12 mini)
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 375, "height": 812}, device_scale_factor=2)
        
        for file in html_files:
            file_path = f"file://{os.path.join(mockups_dir, file)}"
            await page.goto(file_path)
            # Wait for any external resources to load
            await page.wait_for_timeout(1000)
            
            out_file = os.path.join(screenshots_dir, file.replace('.html', '.png'))
            await page.screenshot(path=out_file)
            print(f"Saved screenshot: {out_file}")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(take_screenshots())
