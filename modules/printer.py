# pip install pypiwin32
import win32print
import win32ui
import win32con


def print2Printer():
    INCH = 1440

    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(win32print.GetDefaultPrinter())
    hDC.StartDoc("Test doc")
    hDC.StartPage()
    hDC.SetMapMode(win32con.MM_TWIPS)
    hDC.DrawText("TEST HELLO  WORLD! CORSS FIREWALL, WE TOUCH THE WORLD!",
                 (0, INCH * -1, INCH * 8, INCH * -2), win32con.DT_CENTER)
    hDC.EndPage()
    hDC.EndDoc()


if __name__ == "__main__":
    print2Printer()
