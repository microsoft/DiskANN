#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

enum class YamlNodeType
{
    Undefined,
    Scalar,
    Map,
    Sequence
};

struct YamlNode
{
    YamlNodeType type;
    std::string key;                          // Only used for map nodes
    std::string value;                        // Only used for scalar nodes
    std::map<std::string, YamlNode> mapValue; // Only used for map nodes
    std::vector<YamlNode> sequenceValue;      // Only used for sequence nodes
};

// Function to parse the YAML file and build the node structure
YamlNode ParseYaml(const std::string &filename)
{
    std::ifstream file(filename);
    YamlNode rootNode;

    if (file.is_open())
    {
        std::vector<std::pair<int, YamlNode *>> stack;
        stack.emplace_back(-1, &rootNode);

        std::string line;
        while (std::getline(file, line))
        {
            int indentLevel = 0;
            while (indentLevel < line.size() && line[indentLevel] == ' ')
            {
                indentLevel++;
            }

            line = line.substr(indentLevel);
            if (line.empty())
            {
                continue; // Ignore empty lines
            }

            YamlNode node;
            if (line[0] == '-')
            {
                // Sequence node
                node.type = YamlNodeType::Sequence;
                node.value = line.substr(1);
                stack.back().second->sequenceValue.push_back(node);
            }
            else
            {
                // Map or scalar node
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos)
                {
                    // Map node
                    node.type = YamlNodeType::Map;
                    node.key = line.substr(0, colonPos);
                    auto *parent = stack.back().second;
                    stack.emplace_back(indentLevel, &(parent->mapValue[node.key]));
                    *(stack.back().second) = node;
                }
                else
                {
                    // Scalar node
                    node.type = YamlNodeType::Scalar;
                    node.value = line;
                    stack.back().second->value = node.value;
                }
            }

            // Pop nodes from stack if the indent level decreased
            while (!stack.empty() && stack.back().first >= indentLevel)
            {
                stack.pop_back();
            }
        }

        file.close();
    }
    else
    {
        std::cerr << "Failed to open the YAML file." << std::endl;
    }

    return rootNode;
}

// Function to recursively print the YAML node structure
void PrintYamlNode(const YamlNode &node, const std::string &parentKey = "", int indentLevel = 0)
{
    std::string indent(indentLevel, ' ');

    if (!parentKey.empty())
    {
        std::cout << indent << parentKey << ":" << std::endl;
    }

    if (node.type == YamlNodeType::Scalar)
    {
        std::cout << indent << "  " << node.key << ": " << node.value << std::endl;
    }
    else if (node.type == YamlNodeType::Map)
    {
        for (const auto &entry : node.mapValue)
        {
            PrintYamlNode(entry.second, entry.first, indentLevel + 2);
        }
    }
    else if (node.type == YamlNodeType::Sequence)
    {
        for (const auto &entry : node.sequenceValue)
        {
            PrintYamlNode(entry, "", indentLevel);
        }
    }
}