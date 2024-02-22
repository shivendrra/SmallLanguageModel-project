const fs = require('fs');
const axios = require('axios');
const cheerio = require('cheerio');

const jsonData = require('../../data/json outputs/britannica_links.json');
const outFile = '../../data/scrapped files/raw_output.txt';
let n_res = 0;

start_time = Date.now();

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
    n_res = n_res + 1;
    console.log(`${url}'s text data saved to ${fileName}`);
  }
  catch (error) {
    console.error('Error saving data to file:', error.message);
  }
}

async function main() {
  for (const queryData of jsonData) {
    for (const linkSet of queryData.links) {
      for (const link of linkSet) {
        const url = `https://www.britannica.com${link}`;
        const htmlData = await fetchData(url);
        if (htmlData) {
          const textData = extractTextFromHtml(htmlData);
          await saveTextToFile(textData, outFile, url);
        }
      }
    }
  }
  end_time = Date.now();
  console.log(`total ${n_res} results fetched`);
  console.log(`total time taken ${(end_time - start_time) / (60 * 1000)} mins`);
}

main();