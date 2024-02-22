const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('https://cse.google.com/cse?cx=27d4bab791cce4187#gsc.tab=0&gsc.q=Wikipedia&gsc.sort=&gsc.page=1');
  await page.waitForTimeout(3000);

  const html = await page.content();
  fs.appendFileSync('../faaltu.html', html)
  await browser.close();
})();