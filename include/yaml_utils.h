#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

class YAMLNode
{
  public:
    YAMLNode()
    {
    }

    YAMLNode(const std::string &key, const std::string &value) : key(key), val(value)
    {
    }

    YAMLNode &get(const std::string &key)
    {
        return children[key];
    }

    const std::string &value() const
    {
        return val;
    }

  private:
    std::string key;
    std::string val;
    std::map<std::string, YAMLNode> children;
    friend class YAMLParser;
};

class YAMLParser
{
  public:
    YAMLParser(const std::string &filePath) : filePath(filePath)
    {
    }

    bool parse()
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            std::cout << "Failed to open file: " << filePath << std::endl;
            return false;
        }

        std::string line;
        std::string key;
        std::string value;
        std::vector<std::string> keyStack;
        YAMLNode *currentNode = nullptr;

        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            std::string token;
            std::getline(iss, token, ' ');

            if (token.empty())
            {
                continue;
            }
            else if (token == "-")
            {
                keyStack.pop_back();
            }
            else if (token.back() == ':')
            {
                key = token.substr(0, token.length() - 1);
                keyStack.push_back(key);
                currentNode = &getRootNode();
                for (const auto &parentKey : keyStack)
                {
                    currentNode = &currentNode->get(parentKey);
                }
            }
            else
            {
                value = token;
                currentNode->get(key).val = value;
            }
        }

        return true;
    }

    YAMLNode &getRootNode()
    {
        return rootNode;
    }

  private:
    std::string filePath;
    YAMLNode rootNode;
};