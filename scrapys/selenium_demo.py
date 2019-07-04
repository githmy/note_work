#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium.webdriver.common.keys import Keys
from selenium.webdriver import Chrome
import time
import random
import json
import pandas as pd
import os
import copy


# In[2]:


class Yangcong():
    def __init__(self):
        self.url = "https://yangcong345.com/#/login?type=login"
        self.driver = Chrome("D:\Chrome下载\chromedriver.exe")

    def log_in(self):
        self.driver.get(self.url)
        time.sleep(random.randint(3, 5))  # 睡3分钟，等待页面加载
        #         self.driver.save_screenshot("0.jpg")
        # 输入账号
        self.driver.find_element_by_xpath('//*[@id="username"]').send_keys("18721986267")
        # 输入密码
        self.driver.find_element_by_xpath('//*[@id="password"]').send_keys("fff111QQQ")
        # 点击登陆
        self.driver.find_element_by_class_name("btn-bg-blue").click()
        time.sleep(random.randint(5, 10))

    #         self.driver.save_screenshot("save.jpg")
    # 输出登陆之后的cookies

    def __del__(self):
        self.driver.close()
        # # 关闭浏览器
        # driver.quit()


# In[3]:


yangcong = Yangcong()
yangcong.log_in()  # 之后调用登陆方法

# In[4]:


# # yangcong.driver.find_elements_by_xpath('//li/button')
# aa = yangcong.driver.find_elements_by_xpath('//ul/li/button')


# In[5]:


# for i1 in aa:
#     print(i1.text)


# In[6]:


cookiejson = yangcong.driver.get_cookies()
print(cookiejson)
wstr = cookiejson[0]["value"]
print(wstr)


# In[7]:


# # log print
# def getResponseHeaders(browser):
#     har = json.loads(browser.get_log('har')[0]['message'])
#     return OrderedDict(sorted([(header["name"], header["value"]) for header in har['log']['entries'][0]['response']["headers"]], key = lambda x: x[0]))

# def getResponseStatus(browser):
#     har = json.loads(browser.get_log('har')[0]['message'])
#     return (har['log']['entries'][0]['response']["status"],\
#             str(har['log']['entries'][0]['response']["statusText"]))

# print("status: ", getResponseStatus(yangcong.driver))
# headers = getResponseHeaders(yangcong.driver)
# for key in headers:
#     print(key, "=>", headers[key])


# In[8]:


# # 运行js
# # ajax_url = "https://school-api.yangcong345.com/course/course-tree/themes/{}".format(wstr)
# # ajax_url = "https://api-v5-0.yangcong345.com/progresses?subjectId=1&publisherId=1&semesterId=13&stageId=2"
# ajax_url = '''var n=this;!function(){var t=function(t,e){if(!(t instanceof e))throw new TypeError("Cannot call a class as a function")},e=function(){function t(t,e){for(var n=0;n<e.length;n++){var r=e[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}return function(e,n,r){return n&&t(e.prototype,n),r&&t(e,r),e}}(),r=function(t,e){if(!t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!e||"object"!==typeof e&&"function"!==typeof e?t:e},i=function(){var t=[[[],[],[],[],[]],[[],[],[],[],[]]],e=t[0],n=t[1],r=e[4],i=n[4],a=void 0,o=void 0,s=void 0,u=[],l=[],c=void 0,h=void 0,f=void 0,p=void 0,d=void 0;for(a=0;a<256;a++)l[(u[a]=a<<1^283*(a>>7))^a]=a;for(o=s=0;!r[o];o^=c||1,s=l[s]||1)for(f=(f=s^s<<1^s<<2^s<<3^s<<4)>>8^255&f^99,r[o]=f,i[f]=o,d=16843009*u[h=u[c=u[o]]]^65537*h^257*c^16843008*o,p=257*u[f]^16843008*f,a=0;a<4;a++)e[a][o]=p=p<<24^p>>>8,n[a][f]=d=d<<24^d>>>8;for(a=0;a<5;a++)e[a]=e[a].slice(0),n[a]=n[a].slice(0);return t},a=null,o=function(){function e(n){t(this,e),a||(a=i()),this._tables=[[a[0][0].slice(),a[0][1].slice(),a[0][2].slice(),a[0][3].slice(),a[0][4].slice()],[a[1][0].slice(),a[1][1].slice(),a[1][2].slice(),a[1][3].slice(),a[1][4].slice()]];var r=void 0,o=void 0,s=void 0,u=void 0,l=void 0,c=this._tables[0][4],h=this._tables[1],f=n.length,p=1;if(4!==f&&6!==f&&8!==f)throw new Error("Invalid aes key size");for(u=n.slice(0),l=[],this._key=[u,l],r=f;r<4*f+28;r++)s=u[r-1],(r%f===0||8===f&&r%f===4)&&(s=c[s>>>24]<<24^c[s>>16&255]<<16^c[s>>8&255]<<8^c[255&s],r%f===0&&(s=s<<8^s>>>24^p<<24,p=p<<1^283*(p>>7))),u[r]=u[r-f]^s;for(o=0;r;o++,r--)s=u[3&o?r:r-4],l[o]=r<=4||o<4?s:h[0][c[s>>>24]]^h[1][c[s>>16&255]]^h[2][c[s>>8&255]]^h[3][c[255&s]]}return e.prototype.decrypt=function(t,e,n,r,i,a){var o=this._key[1],s=t^o[0],u=r^o[1],l=n^o[2],c=e^o[3],h=void 0,f=void 0,p=void 0,d=o.length/4-2,g=void 0,m=4,v=this._tables[1],y=v[0],b=v[1],x=v[2],_=v[3],w=v[4];for(g=0;g<d;g++)h=y[s>>>24]^b[u>>16&255]^x[l>>8&255]^_[255&c]^o[m],f=y[u>>>24]^b[l>>16&255]^x[c>>8&255]^_[255&s]^o[m+1],p=y[l>>>24]^b[c>>16&255]^x[s>>8&255]^_[255&u]^o[m+2],c=y[c>>>24]^b[s>>16&255]^x[u>>8&255]^_[255&l]^o[m+3],m+=4,s=h,u=f,l=p;for(g=0;g<4;g++)i[(3&-g)+a]=w[s>>>24]<<24^w[u>>16&255]<<16^w[l>>8&255]<<8^w[255&c]^o[m++],h=s,s=u,u=l,l=c,c=h},e}(),s=function(){function e(){t(this,e),this.listeners={}}return e.prototype.on=function(t,e){this.listeners[t]||(this.listeners[t]=[]),this.listeners[t].push(e)},e.prototype.off=function(t,e){if(!this.listeners[t])return!1;var n=this.listeners[t].indexOf(e);return this.listeners[t].splice(n,1),n>-1},e.prototype.trigger=function(t){var e=this.listeners[t];if(e)if(2===arguments.length)for(var n=e.length,r=0;r<n;++r)e[r].call(this,arguments[1]);else for(var i=Array.prototype.slice.call(arguments,1),a=e.length,o=0;o<a;++o)e[o].apply(this,i)},e.prototype.dispose=function(){this.listeners={}},e.prototype.pipe=function(t){this.on("data",function(e){t.push(e)})},e}(),u=function(e){function n(){t(this,n);var i=r(this,e.call(this,s));return i.jobs=[],i.delay=1,i.timeout_=null,i}return function(t,e){if("function"!==typeof e&&null!==e)throw new TypeError("Super expression must either be null or a function, not "+typeof e);t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,enumerable:!1,writable:!0,configurable:!0}}),e&&(Object.setPrototypeOf?Object.setPrototypeOf(t,e):t.__proto__=e)}(n,e),n.prototype.processJob_=function(){this.jobs.shift()(),this.jobs.length?this.timeout_=setTimeout(this.processJob_.bind(this),this.delay):this.timeout_=null},n.prototype.push=function(t){this.jobs.push(t),this.timeout_||(this.timeout_=setTimeout(this.processJob_.bind(this),this.delay))},n}(s),l=function(t){return t<<24|(65280&t)<<8|(16711680&t)>>8|t>>>24},c=function(){function n(e,r,i,a){t(this,n);var o=n.STEP,s=new Int32Array(e.buffer),c=new Uint8Array(e.byteLength),h=0;for(this.asyncStream_=new u,this.asyncStream_.push(this.decryptChunk_(s.subarray(h,h+o),r,i,c)),h=o;h<s.length;h+=o)i=new Uint32Array([l(s[h-4]),l(s[h-3]),l(s[h-2]),l(s[h-1])]),this.asyncStream_.push(this.decryptChunk_(s.subarray(h,h+o),r,i,c));this.asyncStream_.push(function(){var t;a(null,(t=c).subarray(0,t.byteLength-t[t.byteLength-1]))})}return n.prototype.decryptChunk_=function(t,e,n,r){return function(){var i=function(t,e,n){var r=new Int32Array(t.buffer,t.byteOffset,t.byteLength>>2),i=new o(Array.prototype.slice.call(e)),a=new Uint8Array(t.byteLength),s=new Int32Array(a.buffer),u=void 0,c=void 0,h=void 0,f=void 0,p=void 0,d=void 0,g=void 0,m=void 0,v=void 0;for(u=n[0],c=n[1],h=n[2],f=n[3],v=0;v<r.length;v+=4)p=l(r[v]),d=l(r[v+1]),g=l(r[v+2]),m=l(r[v+3]),i.decrypt(p,d,g,m,s,v),s[v]=l(s[v]^u),s[v+1]=l(s[v+1]^c),s[v+2]=l(s[v+2]^h),s[v+3]=l(s[v+3]^f),u=p,c=d,h=g,f=m;return a}(t,e,n);r.set(i,t.byteOffset)}},e(n,null,[{key:"STEP",get:function(){return 32e3}}]),n}();new function(t){t.onmessage=function(e){var n=e.data,r=new Uint8Array(n.encrypted.bytes,n.encrypted.byteOffset,n.encrypted.byteLength),i=new Uint32Array(n.key.bytes,n.key.byteOffset,n.key.byteLength/4),a=new Uint32Array(n.iv.bytes,n.iv.byteOffset,n.iv.byteLength/4);new c(r,i,a,function(e,r){t.postMessage(function(t){var e={};return Object.keys(t).forEach(function(n){var r=t[n];ArrayBuffer.isView(r)?e[n]={bytes:r.buffer,byteOffset:r.byteOffset,byteLength:r.byteLength}:e[n]=r}),e}({source:n.source,decrypted:r}),[r.buffer])})}}(n)}()'''
# js = 'window.location.href="{}"'.format(ajax_url)
# yangcong.driver.execute_script(js)


# In[9]:


def get3keys():
    return yangcong.driver.find_elements_by_xpath('//div[@class="select-text"]/div[contains(@class,"select")]')


def get_points3():
    return yangcong.driver.find_elements_by_xpath('//h3[contains(@class,"exhibit-son")]')


# def get_button_mother():
#     return yangcong.driver.find_elements_by_xpath('//div[contains(@class,"wrape-mid")]//ul//button')
# aa = get_button_mother()
# print(aa)
# print(len(aa))
def get_button():
    return yangcong.driver.find_elements_by_xpath('//div[contains(@class,"wrape-main")]//ul//button')


def get_button_back():
    return yangcong.driver.find_element_by_xpath('//button')


# In[13]:


keylist0 = yangcong.driver.find_elements_by_xpath('//div[@class="select-text"]/div[contains(@class,"select")]')
keylist0 = keylist0[0]
keylist0.click()
time.sleep(3)
llt1 = keylist0.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
print("len_press" + str(len(llt1)))
for i1 in range(len(llt1)):
    # 内部是出版社
    yangcong.driver.get(urll1)
    #     yangcong.driver.refresh()
    #     time.sleep(5)
    #     print("back_url1")
    #     print(urll1)
    print(i1)
    keylist0 = yangcong.driver.find_elements_by_xpath('//div[@class="select-text"]/div[contains(@class,"select-wrap")]')
    keylist0 = keylist0[0]
    keylist0.click()
    time.sleep(3)
    ll1 = keylist0.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
    print("len_press" + str(len(ll1)))
    print(ll1[i1])
    ll1[i1].click()
    time.sleep(3)
    s_press = ll1[i1].find_element_by_xpath('../../../..//div[@class="chosen"]').text
    print(s_press)

# In[11]:


json_data = []

urll1 = yangcong.driver.current_url

keylist0 = get3keys()[0]
keylist0.click()
time.sleep(random.randint(1, 3))
llt1 = keylist0.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
print("len_press" + str(len(llt1)))
for i1 in range(len(llt1)):
    # 内部是出版社
    yangcong.driver.get(urll1)
    time.sleep(random.randint(3, 5))
    print("back_url1")
    print(urll1)
    keylist0 = get3keys()[0]
    ll1 = keylist0.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
    keylist0.click()
    keylist0.click()
    print("len_press" + str(len(ll1)))
    ll1[i1].click()
    time.sleep(random.randint(3, 5))
    s_press = ll1[i1].find_element_by_xpath('../../../..//div[@class="chosen"]').text
    print(s_press)
    keylist1 = get3keys()[1]
    keylist1.click()
    time.sleep(random.randint(1, 3))
    ll2 = keylist1.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
    print("len_grade" + str(len(ll2)))
    for i2 in range(len(ll2)):
        # 内部是年级
        yangcong.driver.get(urll1)
        time.sleep(random.randint(3, 5))
        print("back_url1")
        keylist1 = get3keys()[1]
        #         keylist1.click()
        #         keylist1.click()
        ll2 = keylist1.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
        print("len_grade" + str(len(ll2)))
        ll2[i2].click()
        time.sleep(random.randint(3, 5))
        s_grade = ll2[i2].find_element_by_xpath('../../../..//div[@class="chosen"]').text
        print(s_grade)
        keylist2 = get3keys()[2]
        keylist2.click()
        time.sleep(random.randint(1, 3))
        ll3 = keylist2.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
        print("len_chapter" + str(len(ll3)))
        for i3 in range(len(ll3)):
            # 内部是章节
            yangcong.driver.get(urll1)
            time.sleep(random.randint(3, 5))
            print("back_url1")
            print(urll1)
            keylist2 = get3keys()[2]
            #             keylist2.click()
            #             keylist2.click()
            #             print(keylistt[2].find_elements_by_xpath('.//ul[@class="select-value show"]/li/button'))
            #             time.sleep(random.randint(1,3))
            ll3 = keylist2.find_elements_by_xpath('.//ul[@class="select-value show"]/li/button')
            print("len_chapter" + str(len(ll3)))
            ll3[i3].click()
            time.sleep(random.randint(3, 5))
            s_chapter = ll3[i3].find_element_by_xpath('../../..//div[@class="chosen"]').text
            print(s_chapter)
            points3 = get_points3()
            time.sleep(random.randint(1, 3))
            print("len_point " + str(len(points3)))
            for i4 in range(len(points3)):
                # 内部是考点
                yangcong.driver.get(urll1)
                print("back_url1")
                time.sleep(random.randint(3, 5))
                points3 = get_points3()
                print("len_point " + str(len(points3)))
                s_points = points3[i4].text
                print(s_points)
                ll4 = points3[i4].find_elements_by_xpath('..//h4')
                print("len_type " + str(len(ll4)))

                for i5 in range(len(ll4)):
                    # 内部是 概念或解题
                    yangcong.driver.get(urll1)
                    time.sleep(random.randint(3, 5))
                    print("back_url1")
                    points3 = get_points3()
                    ll4 = points3[i4].find_elements_by_xpath('..//h4')
                    print("len_point " + str(len(points3)))
                    print("len_type " + str(len(ll4)))
                    s_type = ll4[i5].text
                    print(s_type)
                    ll4[i5].click()
                    time.sleep(random.randint(3, 5))
                    urll2 = yangcong.driver.current_url
                    buttons = get_button()
                    print("len_button " + str(len(buttons)))
                    for i6 in range(len(buttons)):
                        # 内部是单个视频
                        yangcong.driver.get(urll2)
                        print("back_url2")
                        time.sleep(random.randint(3, 5))
                        buttons = get_button()
                        print("len_button " + str(len(buttons)))
                        time.sleep(random.randint(1, 3))
                        s_deep_point = buttons[i6].find_element_by_xpath('..//h3').text
                        print(s_deep_point)
                        nosig = False
                        try:
                            buttons[i6].click()
                            time.sleep(random.randint(1, 3))
                            tbt = yangcong.driver.find_element_by_xpath('//i[@class="svgchapter video"]')
                            time.sleep(random.randint(3, 5))
                            if tbt is not None:
                                tbt.click()
                                print("have")
                            else:
                                tbt = yangcong.driver.find_element_by_xpath('//div[@class="alert-ft"]/button')
                                tbt.click()
                                nosig = True
                        except Exception as e:
                            print(e)
                        s_url = yangcong.driver.current_url
                        if True == nosig:
                            s_url = "需要付费"
                            print("not have")
                        print(s_url)
                        # 保存
                        tmpjson = {
                            "s_press": s_press,
                            "s_grade": s_grade,
                            "s_chapter": s_chapter,
                            "s_points": s_points,
                            "s_type": s_type,
                            "s_url": s_url,
                            "s_deep_point": s_deep_point,
                            "seed_file": ""
                        }
                        print(tmpjson)
                        json_data.append(tmpjson)
                        time.sleep(random.randint(3, 10))
                        #                         yangcong.driver.back()
                        #                         time.sleep(random.randint(3,5))
                        print("back")
                    #                     back_points = get_button_back()
                    #                     yangcong.driver.back()
                    #                     time.sleep(random.randint(3,5))
                    #             yangcong.driver.executeScript("window.scrollTo(0,0)")
                cmd_path = os.getcwd()
                basic_path = os.path.join(cmd_path, "..", "data")
                project_path = os.path.join(basic_path, "spider")
                data_path = os.path.join(project_path, "data")
                save_csv = os.path.join(data_path, "yangcong.csv")
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                savepd = pd.DataFrame(json_data)
                savepd.to_csv(save_csv, encoding='utf-8', index=False)
                print("one end")

