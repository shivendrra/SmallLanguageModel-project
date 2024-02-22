// it fetches and scrappes links from the Google Custom Search engine's hosted outputs

const fs = require('fs');
const axios = require('axios');
const cheerio = require('cheerio');

const dotenv = require('dotenv');
const path = require('path');
const dotenvPath = path.resolve('../', '.env');
dotenv.config({ path: dotenvPath });

const api_key = process.env.search_key;
const cx_id = process.env.cx_id;

const file_path = '../data/json outputs/search_strings.json';
const search_strings = JSON.parse(fs.readFileSync(file_path, 'utf-8'));
const outputFileName = '../data/json outputs/output.json';
let n_results = 0;
let pageNo = 1

const start_time = Date.now();

async function gSearch(searchTerm, cxId, pageNo) {
  const formattedSearchTerm = searchTerm.split(' ').join('%');
  const url = `https://cse.google.com/cse?cx=${cxId}#gsc.tab=0&gsc.q=${formattedSearchTerm}&gsc.sort=&gsc.page=${pageNo}`;
  const response = await axios.get(url);
  return response;
}

async function scrapeLinks(htmlContent) {
  const $ = cheerio.load(htmlContent);
  const links = $('a.gs-title').map((index, element) => $(element).attr('href')).get();
  const actualLinks = links.map(link => new URLSearchParams(link).get('q'));
  return actualLinks;
}

async function main(fileName, pgNo) {
  for (const searchTerm of search_strings) {
    const response = await gSearch(searchTerm, cx_id, pgNo);
    if (response.status === 200 && pgNo < 11) {
      const links = await scrapeLinks(response.data);
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