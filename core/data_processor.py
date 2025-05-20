# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Dict
from llama_index.core.schema import TextNode


class DataProcessor:
    """数据处理工具类"""

    @staticmethod
    def load_and_validate_json(data_dir: str) -> List[Dict]:
        """加载并验证JSON法律文件"""
        json_files = list(Path(data_dir).glob("*.json"))
        assert json_files, f"未找到JSON文件于 {data_dir}"

        all_data = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")

                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")

                all_data.extend({
                                    "content": item,
                                    "metadata": {"source": json_file.name}
                                } for item in data)

        print(f"成功加载 {len(all_data)} 个法律文件条目")
        return all_data

    @staticmethod
    def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
        """创建带稳定ID的文本节点"""
        nodes = []
        for entry in raw_data:
            law_dict = entry["content"]
            source_file = entry["metadata"]["source"]

            for full_title, content in law_dict.items():
                node_id = f"{source_file}::{full_title}"
                parts = full_title.split(" ", 1)

                node = TextNode(
                    text=content,
                    id_=node_id,
                    metadata={
                        "law_name": parts[0] if parts else "未知法律",
                        "article": parts[1] if len(parts) > 1 else "未知条款",
                        "full_title": full_title,
                        "source_file": source_file,
                        "content_type": "legal_article"
                    }
                )
                nodes.append(node)

        print(f"生成 {len(nodes)} 个文本节点（示例ID：{nodes[0].id_}）")
        # print(f"生成 {len(nodes)} 个文本节点（示例ID：{nodes[:3]}）")
        return nodes