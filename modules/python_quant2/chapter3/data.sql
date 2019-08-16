/*
Navicat MySQL Data Transfer

Source Server         : test
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : test

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2018-03-11 18:17:42
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for data
-- ----------------------------
DROP TABLE IF EXISTS `data`;
CREATE TABLE `data` (
  `ID` varchar(100) DEFAULT NULL,
  `stockname` varchar(100) DEFAULT NULL,
  `price` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of data
-- ----------------------------
INSERT INTO `data` VALUES ('1', '格力电器', '10');
INSERT INTO `data` VALUES ('2', '中国平安', '34');
INSERT INTO `data` VALUES ('3', '浦发银行', '16');
INSERT INTO `data` VALUES ('4', '万科A', '25');
