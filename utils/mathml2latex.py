import os
from lxml import etree


def mathml2latex_yarosh(equation):
    """ MathML to LaTeX conversion with XSLT from Vasil Yaroshevich """

    xslt_file = os.path.join("C:\\", "project", "data", "xsltml_2.0", "mmltex.xsl")
    dom = etree.fromstring(equation)
    xslt = etree.parse(xslt_file)
    transform = etree.XSLT(xslt)
    newdom = transform(dom)
    return newdom


def main():
    # baspath = os.path.join("D:\\", "Chrome下载", "mnist_dataset")
    xslt_file = os.path.join("C:\\", "project", "data", "Minor-master", "crohme_dataset", "truth", "algb09.txt")
    mathml = """<math xmlns="http://www.w3.org/1998/Math/MathML">
      <mrow>
        <mfrac>
          <mrow><mi>x</mi></mrow>
          <mrow><mi>y</mi></mrow>
        </mfrac>
      </mrow>
    </math>"""
    tex = mathml2latex_yarosh(mathml)
    print(tex)


if __name__ == '__main__':
    main()
