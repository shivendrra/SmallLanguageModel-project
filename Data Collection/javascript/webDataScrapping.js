// scrappes data from the links fetched and saved by the Google Custom Search Engnine

const fs = require('fs');
const axios = require('axios');
const cheerio = require('cheerio');

const jsonData = fs.readFileSync('../../data/json outputs/search_results_v2.json');
const searchResults = JSON.parse(jsonData);
const outputFileName = '../../data/scrapped files/output.txt';
let n_pages = 0;

async function fetchData(url, timeout = 10000) {
  try {
    const response = await axios.get(url, {timeout});
    return response.data;
  } 
  catch (error) {
    if(axios.isCancel(error)){
      console.error(`Timeout fetching data from ${url} (exceeded ${timeout} milliseconds)`);
    } else {
    console.error(`Error fetching data from ${url}:`, error.message);
    }
    return null;
  }
}

function extractTextFromHtml(html) {
  const $ = cheerio.load(html);
  const headings = $('h1', 'h2', 'h3', 'h4', 'h5').map((index, element) => $(element).text()).get();
  const paragraphs = $('p').map((index, element) => $(element).text()).get();
  return headings.join('\n'), paragraphs.join('\n');
}

async function saveTextToFile(text, fileName, url) {
  try {
    fs.appendFileSync(fileName, text);
    n_pages = n_pages + 1;
    console.log(`${url}'s text data saved to ${fileName}`);
  }
  catch (error) {
    console.error('Error saving data to file:', error.message);
  }
}

async function main() {
  for (const result of searchResults) {
    const htmlData = await fetchData(result.link);
    if (htmlData) {
      const textData = extractTextFromHtml(htmlData);
      await saveTextToFile(textData, outputFileName, result.link);
    }
  }
  console.log(`total ${n_pages}`);
}

main();