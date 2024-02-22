// some errors

const fs = require('fs');
const puppeteer = require('puppeteer');
const axios = require('axios');
const cheerio = require('cheerio');

const dotenv = require('dotenv');
const path = require('path');
const dotenvPath = path.resolve('../', '.env');
dotenv.config({ path: dotenvPath });

const api_key = process.env.search_key;
const cx_id = process.env.cx_id;
const file_path = '../data/search_strings.json';
const search_strings = JSON.parse(fs.readFileSync(file_path, 'utf-8'));
const outputFileName = '../data/output.json';
let n_results = 0;
let pageNo = 1

const start_time = Date.now();

async function scrapeContent(url) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  // Navigate to the URL
  await page.goto(url);
  const content = await page.content();
  fs.appendFileSync('../faaltu.html', content)
  await browser.close();
  return content;
}


async function scrapeLinks(htmlContent) {
  const $ = cheerio.load(htmlContent);
  const links = $('a.gs-title').map((index, element) => $(element).attr('href')).get();
  // const actualLinks = links.map(link => new URLSearchParams(link).get('q'));
  return links;
}

async function gSearch(searchTerm, cxId, pageNo) {
  const formattedSearchTerm = searchTerm.split(' ').join('%');
  const url = `https://cse.google.com/cse?cx=${cxId}#gsc.tab=0&gsc.q=${formattedSearchTerm}&gsc.sort=&gsc.page=${pageNo}`;
  return url;
}

async function main(fileName, pgNo) {
  for (const searchTerm of search_strings) {
    const url = await gSearch(searchTerm, cx_id, pgNo);
    if (pgNo < 11) {
      const scrape = await scrapeContent(url);
      const links = await scrapeLinks(scrape);
      n_results += 10;
      pgNo += 1;
      fs.appendFileSync(fileName, JSON.stringify({ searchTerm, links }) + '\n');
    } else {
      console.error(`Error in search for '${searchTerm}': ${response.status}`);
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  console.log(`Total ${n_results} results`);
}

main(outputFileName, pageNo).then(() => {
  const end_time = Date.now();
  console.log(`Time taken to fetch and process results: ${(end_time - start_time) / 1000} seconds`);
});