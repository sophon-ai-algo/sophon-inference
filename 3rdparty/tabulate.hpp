#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

class Table {
  //Base data
  std::vector<std::string> _head;
  std::vector<std::vector<std::string>> _body;
  //Format data
  int num_row;
  int num_col;
  std::string headStr;
  std::vector<int> colWidths;
  std::vector<std::string> rowStrs;
 public:
  //Functions
  void setName(const std::string& name);
  void setHead(const std::vector<std::string>& head);
  void setBody(const std::vector<std::vector<std::string>>& body);
  void print();
};

void Table::setHead(const std::vector<std::string>& head) {
  if (!_body.empty()) {
    printf("\033[1;31m");
    printf("Table not empty, can't set header\n");
    printf("\033[0m");
    exit(-1);
  }
  _head.insert(_head.end(), head.begin(), head.end());
  num_col = _head.size();
}

void Table::setBody(const std::vector<std::vector<std::string>>& body) {
  std::for_each(body.begin(), body.end(), [&](auto item){
    if (item.size() != _head.size()) {
      printf("\033[1;31m");
      printf("Length of row not the same as header\n");
      printf("\033[0m");
      exit(-1);
    }
  });
  _body.insert(_body.end(), body.begin(), body.end());
  num_row = _body.size();
}

void Table::print() {
  if (_head.empty() || _body.empty()) {
    printf("\033[1;31m");
    printf("Missing data, num_col %d, num_row %d\n", num_col, num_row); 
    printf("\033[0m");
    exit(-1);
  }
  // Find max width of each column
  std::transform(_head.begin(), _head.end(), std::back_inserter(colWidths),
                 [](auto item){ return item.size(); });
  for (size_t r = 0; r < _body.size(); r++) {
    for (size_t c = 0; c < _head.size(); c++) {
      colWidths[c] = std::max(int(_body[r][c].length()),colWidths[c]);
    }
  }
  // +3 because it includes the space for the first |
  // and a space on each side of the column name
  // +1 to make space for the last |
  int totalWidth = std::accumulate(colWidths.begin(), colWidths.end(), 0,
                   [](int sum, auto item){ return sum + item + 3; }) + 1;
  const int lengthDiff = 4;
  const int numPreSpace = 2;
  const int numPostSpace = 2;
  const std::string preStr(numPreSpace, ' ');
  const std::string postStr(numPostSpace, ' ');
  // Create string with each column name
  for (size_t i = 0; i < num_col; i++) {
    const std::string& col = _head[i];
    const int lengthDiff = colWidths[i] - col.length();
    if (lengthDiff > 0) {
      // Divide by 2 to get number of pre spaces
      const int numPreSpace = lengthDiff / 2;
      // Divide by 2, but increment lengthDiff by one to round up.
      // I do this because I want extra spaces after the column name
      const int numPostSpace = (lengthDiff + 1) / 2;
      const std::string preStr(numPreSpace, ' ');
      const std::string postStr(numPostSpace, ' ');
      headStr += "| " + preStr + col + postStr + " ";
    } else {
      headStr += "| " + col + " ";
    }
  }
  headStr += "|";
  // Create string for each row and its elements
  rowStrs = std::vector<std::string>(num_row);
  for (size_t r = 0; r < num_row; r++) {
    for (size_t e = 0; e < _body[r].size(); e++) {
      const std::string& elementData = _body[r][e];
      const int lengthDiff = colWidths[e] - elementData.length();
      // Divide by 2 to get number of pre spaces
      const int numPreSpace = lengthDiff / 2;
      // Divide by 2, but increment lengthDiff by one to round up.
      // I do this because I want extra spaces after the column name
      const int numPostSpace = (lengthDiff + 1) / 2;
      const std::string preStr(numPreSpace, ' ');
      const std::string postStr(numPostSpace, ' ');
      rowStrs[r] += "| " + preStr + elementData + postStr + " ";
    }
    rowStrs[r] += "|";
  }

  std::string lineThinStr = std::accumulate(colWidths.begin(), colWidths.end(), std::string("+"),
                            [](std::string str, auto item){ return str+std::string(item+2,'-')+"+";});
  std::string lineBoldStr = std::accumulate(colWidths.begin(), colWidths.end(), std::string("+"),
                            [](std::string str, auto item){ return str+std::string(item+2,'=')+"+";});
  // Print table
  printf("\033[0;32m");
  printf("%s\n", lineThinStr.c_str());
  printf("%s\n", headStr.c_str());
  printf("%s\n", lineBoldStr.c_str());
  for (const std::string& rowStr : rowStrs) {
    printf("%s\n", rowStr.c_str());
    printf("%s\n", lineThinStr.c_str());
  }
  printf("\033[0m");
}
