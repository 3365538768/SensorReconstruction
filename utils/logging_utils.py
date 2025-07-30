#!/usr/bin/env python3
"""
日志工具模块
统一管理4DGaussians和笼节点模型的训练记录
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, 
                 log_type: str,  # "4DGaussians" or "cage_model"
                 experiment_name: str,
                 log_dir: str = "logs"):
        """
        初始化日志记录器
        
        Args:
            log_type: 日志类型，"4DGaussians" 或 "cage_model"
            experiment_name: 实验名称
            log_dir: 日志根目录
        """
        self.log_type = log_type
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        
        # 创建时间戳
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置日志文件路径
        self.log_subdir = self.log_dir / log_type / experiment_name
        self.log_subdir.mkdir(parents=True, exist_ok=True)
        
        # 训练日志文件
        self.training_log_file = self.log_subdir / f"training_{self.timestamp}.log"
        
        # 配置日志文件
        self.config_file = self.log_subdir / f"config_{self.timestamp}.json"
        
        # 性能统计文件
        self.metrics_file = self.log_subdir / f"metrics_{self.timestamp}.json"
        
        # 设置Python logging
        self._setup_logger()
        
        # 初始化统计数据
        self.metrics = {
            "start_time": datetime.datetime.now().isoformat(),
            "log_type": log_type,
            "experiment_name": experiment_name,
            "training_stats": {},
            "performance_metrics": {}
        }
        
    def _setup_logger(self):
        """设置Python日志记录器"""
        # 创建logger
        self.logger = logging.getLogger(f"{self.log_type}_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 文件handler
        file_handler = logging.FileHandler(self.training_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        config_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "log_type": self.log_type,
            "experiment_name": self.experiment_name,
            "config": config
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"配置已保存到: {self.config_file}")
        
    def log_training_start(self, **kwargs):
        """记录训练开始"""
        start_info = {
            "start_time": datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        self.metrics["training_stats"]["start_info"] = start_info
        self.logger.info(f"🚀 {self.log_type} 训练开始")
        self.logger.info(f"实验名称: {self.experiment_name}")
        
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
            
    def log_epoch_stats(self, epoch: int, **stats):
        """记录每个epoch的统计信息"""
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            **stats
        }
        
        if "epochs" not in self.metrics["training_stats"]:
            self.metrics["training_stats"]["epochs"] = []
            
        self.metrics["training_stats"]["epochs"].append(epoch_data)
        
        # 记录到日志
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        self.logger.info(f"Epoch {epoch} - {stats_str}")
        
    def log_iteration_stats(self, iteration: int, **stats):
        """记录每个iteration的统计信息"""
        if "iterations" not in self.metrics["training_stats"]:
            self.metrics["training_stats"]["iterations"] = []
            
        iter_data = {
            "iteration": iteration,
            "timestamp": datetime.datetime.now().isoformat(),
            **stats
        }
        
        self.metrics["training_stats"]["iterations"].append(iter_data)
        
        # 每100次iteration记录一次详细日志
        if iteration % 100 == 0:
            stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
            self.logger.info(f"Iteration {iteration} - {stats_str}")
            
    def log_training_complete(self, **final_stats):
        """记录训练完成"""
        completion_info = {
            "end_time": datetime.datetime.now().isoformat(),
            "total_duration": None,  # 将在save_metrics中计算
            **final_stats
        }
        
        self.metrics["training_stats"]["completion_info"] = completion_info
        self.logger.info(f"✅ {self.log_type} 训练完成")
        
        for key, value in final_stats.items():
            self.logger.info(f"{key}: {value}")
            
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """记录错误信息"""
        self.logger.error(f"❌ 错误: {error_msg}")
        if exception:
            self.logger.error(f"异常详情: {str(exception)}")
            
    def save_metrics(self):
        """保存性能指标到文件"""
        # 计算总训练时间
        if "start_info" in self.metrics["training_stats"]:
            start_time = datetime.datetime.fromisoformat(
                self.metrics["training_stats"]["start_info"]["start_time"]
            )
            end_time = datetime.datetime.now()
            duration = str(end_time - start_time)
            
            if "completion_info" in self.metrics["training_stats"]:
                self.metrics["training_stats"]["completion_info"]["total_duration"] = duration
            
        self.metrics["save_time"] = datetime.datetime.now().isoformat()
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"性能指标已保存到: {self.metrics_file}")
        
    def get_log_summary(self) -> Dict[str, str]:
        """获取日志摘要信息"""
        return {
            "log_type": self.log_type,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "training_log": str(self.training_log_file),
            "config_file": str(self.config_file), 
            "metrics_file": str(self.metrics_file),
            "log_directory": str(self.log_subdir)
        }


def create_training_logger(log_type: str, experiment_name: str) -> TrainingLogger:
    """
    创建训练日志记录器的便捷函数
    
    Args:
        log_type: "4DGaussians" 或 "cage_model"
        experiment_name: 实验名称
        
    Returns:
        TrainingLogger实例
    """
    return TrainingLogger(log_type, experiment_name)


def backup_sge_logs(job_id: str, experiment_name: str, log_type: str):
    """
    备份SGE作业日志文件
    
    Args:
        job_id: SGE作业ID
        experiment_name: 实验名称
        log_type: 日志类型
    """
    log_dir = Path("logs") / "sge_jobs" / log_type / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 备份.o和.e文件
    for suffix in ['o', 'e']:
        sge_file = f"*.{suffix}{job_id}"
        target_file = log_dir / f"sge_{suffix}_{timestamp}.log"
        
        # 使用shell命令复制文件
        import subprocess
        try:
            subprocess.run(
                f"cp {sge_file} {target_file}", 
                shell=True, 
                check=True
            )
            print(f"✅ SGE日志已备份: {target_file}")
        except subprocess.CalledProcessError:
            print(f"⚠️ 未找到SGE日志文件: {sge_file}")