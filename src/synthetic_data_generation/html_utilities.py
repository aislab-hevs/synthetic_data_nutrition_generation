from typing import List, Dict, Any

class HTML_Table:
    """This classes produces and static html page that visualizes the summary table after data generation.
    """

    def __init__(self,
                 cols: int = 4,
                 rows: List[str] = None) -> None:
        """Constructor method to create HTML_table object able to render an static HTML page with the summary table.

        :param cols: maximum number of columns in the table, defaults to 4
        :type cols: int, optional
        :param rows: List of rows to incorporate in the table, each row is a string containing HTML tags <tr> <th>, defaults to None
        :type rows: List[str], optional
        """
        self.cols = cols
        if rows is not None:
            self.rows = rows
        else:
            self.rows = []

    def add_rows(self, new_rows: List[str]) -> None:
        """Add a list of rows to the HTML table.

        :param new_rows:  List of rows to be added to the HTML table
        :type new_rows: List[str]
        """
        self.rows.extend(new_rows)

    def add_row(self, row: str) -> None:
        """Add one row to the HTML table.

        :param row: row to be added to the table.
        :type row: str
        """
        self.rows.append(row)

    def get_row(self, row_index: int):
        if row_index >= 0 and row_index < len(self.rows):
            return self.rows[row_index]

    def get_row_count(self):
        return len(self.rows)

    def get_rows(self):
        return self.rows

    def set_value(self, row_number: int, dict_parameters: Dict[str, Any]):
        # check index
        if row_number >= 0 and row_number < len(self.rows):
            self.rows[row_number] = self.rows[row_number].format(
                **dict_parameters)

    def _repr_html_(self) -> str:
        return """
    <!DOCTYPE HTML PUBLIC
	 	"-//W3 Organization//DTD W3 HTML 2.0//EN">
    <html>
    <head>
    <title>
    Summary Table
    </title>
    </head>
    <body>
    <p>
    <ul>
    <li>
    Percentage in <font color="red">red color</font> represent the total percentage respect to all users.
    </li>
    <li>
    Percentage in <font color="green">green color</font> represent the percentage respect to health conditions.
    </li>
    </ul>
    </p>
    <table border=\"1\">
        {row}
    </table>
    </body>
    </html>""".format(row="\n".join(self.rows))

    def render(self) -> str:
        """Returns the HTML str that represents the summary table.

        :return: HTML string representing the HTML summary table.
        :rtype: str
        """
        return self._repr_html_()